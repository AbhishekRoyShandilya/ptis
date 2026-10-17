"""
config/settings.py
Application configuration — reads from environment variables or .env file.

Environment selection:
  PTIS_ENV=test  → loads .env.test  (test DB, port 8001, ./data/test)
  PTIS_ENV=prod  → loads .env       (real DB, port 8000, ./data)

Set PTIS_ENV in the shell BEFORE starting the server. The startup scripts
(start_test.bat / start_prod.bat) do this automatically.
"""
from __future__ import annotations

import os
import logging
from typing import List, Optional

from pydantic_settings import BaseSettings

# Determine env file from PTIS_ENV shell variable (evaluated at import time)
_PTIS_ENV = os.getenv("PTIS_ENV", "prod")
_ENV_FILE = ".env.test" if _PTIS_ENV == "test" else ".env"


class Settings(BaseSettings):
    # ── Application ──
    APP_NAME: str = "PTIS"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    PTIS_ENV: str = "prod"          # "test" | "prod" — set via shell, not .env

    # ── Dev bypass (local development only — NEVER set in production) ──
    # When True, get_current_user returns a fake dev user without checking JWT.
    # Pair with VITE_DEV_BYPASS=true in frontend/.env.local
    DEV_AUTH_BYPASS: bool = False

    # ── Strategy sourcing ──
    # When False (default), the built-in "library" strategy templates
    # (is_library: true) are NOT loaded into the StrategyRegistry. This keeps
    # the system purely user-driven: intelligence, co-planner, behavioral
    # matching and the setup scanner know about NO strategies until the user
    # teaches/approves their own. Set True only to demo the prebuilt library.
    LOAD_LIBRARY_STRATEGY_TEMPLATES: bool = False

    # ── Security ──
    PTIS_JWT_SECRET: str = "ptis-dev-secret-change-in-production-2026"
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:5173,http://localhost:5174"

    # ── Observability ──
    SENTRY_DSN: Optional[str] = None

    # ── Database ──
    DATABASE_URL: str = "postgresql://ptis:ptis@localhost:5432/ptis"

    # ── Market ──
    MARKET_TIMEZONE: str = "Asia/Kolkata"
    MARKET_OPEN_TIME: str = "09:15"    # NSE opens 9:15 AM IST
    MARKET_CLOSE_TIME: str = "15:30"   # NSE closes 3:30 PM IST

    # ── Broker API (generic) ──
    BROKER_API_KEY: Optional[str] = None
    BROKER_API_SECRET: Optional[str] = None
    BROKER_API_URL: Optional[str] = None

    # ── Zerodha / Kite Connect ──
    KITE_API_KEY: Optional[str] = None
    KITE_API_SECRET: Optional[str] = None
    KITE_ACCESS_TOKEN: Optional[str] = None
    KITE_REQUEST_TOKEN: Optional[str] = None

    # ── Telegram alerts (scripts/live/notify.py reads these straight from .env.test). Declared because Settings
    # forbids unknown keys: adding them to .env.test without these fields would break every import of config.settings.
    TELEGRAM_BOT_TOKEN: Optional[str] = None
    TELEGRAM_CHAT_ID: Optional[str] = None

    # ── Upstox ──
    UPSTOX_API_KEY: Optional[str] = None
    UPSTOX_API_SECRET: Optional[str] = None
    UPSTOX_ACCESS_TOKEN: Optional[str] = None
    UPSTOX_REDIRECT_URI: str = "http://localhost:8000/api/v1/broker/upstox/callback"
    # ── Upstox SANDBOX app (separate app, separate credentials; used ONLY by the execution layer's sandbox mode).
    # Declared here because Settings forbids unknown env vars: adding UPSTOX_SANDBOX_TOKEN to .env.test without a
    # field broke EVERY import of config.settings (2026-09-23). Never mix these with the live credentials above.
    UPSTOX_SANDBOX_API_KEY: Optional[str] = None
    UPSTOX_SANDBOX_API_SECRET: Optional[str] = None
    UPSTOX_SANDBOX_TOKEN: Optional[str] = None
    UPSTOX_SANDBOX_REDIRECT_URI: Optional[str] = None

    # ── Angel One / SmartAPI ──
    ANGEL_API_KEY: Optional[str] = None
    ANGEL_CLIENT_ID: Optional[str] = None
    ANGEL_PASSWORD: Optional[str] = None
    ANGEL_TOTP_SECRET: Optional[str] = None

    # ── Groww ──
    GROWW_API_KEY: Optional[str] = None
    GROWW_API_SECRET: Optional[str] = None
    GROWW_ACCESS_TOKEN: Optional[str] = None
    # Base32 TOTP seed for the API key. When set, the adapter auto-mints a fresh
    # access token each day via TOTP — no manual daily token paste needed.
    GROWW_TOTP_SECRET: Optional[str] = None
    # Username of the user who owns the Groww account, so synced trades are
    # attributed to them. If unset, the first real (non-dev) user is used.
    GROWW_OWNER_USERNAME: Optional[str] = None

    # ── LLM ──
    # Primary: gemini (handles text + vision, ~10x cheaper than GPT-4o)
    # Alternatives: openai | anthropic | groq | together | openrouter | fireworks | ollama
    # For Llama 3.3 70B (testing/personal): set LLM_PROVIDER + LLM_MODEL + LLM_API_KEY, e.g.
    #   Groq:    LLM_PROVIDER=groq    LLM_MODEL=llama-3.3-70b-versatile               LLM_API_KEY=<groq key>
    #   Together:LLM_PROVIDER=together LLM_MODEL=meta-llama/Llama-3.3-70B-Instruct-Turbo LLM_API_KEY=<together key>
    #   Ollama:  LLM_PROVIDER=ollama  LLM_MODEL=llama3.3   (no key; local server on :11434)
    # base_url auto-resolves per provider; override with LLM_BASE_URL only for a custom endpoint.
    LLM_PROVIDER: str = "gemini"
    LLM_MODEL: str = "gemini-2.0-flash"
    LLM_API_KEY: Optional[str] = None      # unified key (works for any provider)
    # research-only reader hosts (blind_dossier_bt); declared so the strict env loader accepts them. Not used by the app.
    OPENROUTER_API_KEY: Optional[str] = None
    NVIDIA_API_KEY: Optional[str] = None
    LLM_BASE_URL: Optional[str] = None     # custom OpenAI-compatible endpoint (else provider default)
    GOOGLE_API_KEY: Optional[str] = None   # Alias — either works for Gemini
    LLM_TEMPERATURE: float = 0.3
    LLM_FALLBACK_PROVIDER: str = ""
    LLM_FALLBACK_MODEL: str = ""
    LLM_MAX_RETRIES: int = 1
    LLM_TIMEOUT: int = 20

    # ── Coaching ──
    COACHING_DEFAULT_PERSONALITY: str = "balanced"
    COACHING_MAX_CONTEXT_TRADES: int = 20
    COACHING_MAX_CONVERSATION_HISTORY: int = 20
    COACHING_SESSION_TIMEOUT_HOURS: int = 4

    # ── Data directories (overridden in .env.test to ./data/test/...) ──
    TOKEN_STORE_PATH: str = "./data/broker_tokens.json"

    # ── OHLCV lake (local Parquet + DuckDB market-data store) ──
    # Root for the candle/instrument Parquet lake. Test env should override to
    # ./data/test/lake so test + prod never share market data.
    LAKE_PATH: str = "./data/lake"
    # System-level historical market-data provider (NOT the user's execution
    # broker). "upstox" uses Upstox v3 historical; Angel One / yfinance are
    # automatic fallbacks when configured.
    MARKET_DATA_PROVIDER: str = "upstox"

    # ── Vector Store ──
    VECTOR_STORE_TYPE: str = "chroma"     # chroma | faiss
    VECTOR_STORE_PATH: str = "./data/vector_store"

    # ── Risk Defaults ──
    DEFAULT_RISK_PERCENT: float = 1.0
    MAX_RISK_PERCENT: float = 2.0
    MAX_DAILY_LOSS_PERCENT: float = 3.0
    MAX_DAILY_TRADES: int = 10

    # ── Behavioral Thresholds ──
    REVENGE_TRADE_WINDOW_MINUTES: int = 15
    FOMO_RAPID_ENTRY_MINUTES: int = 5
    OVERSIZE_THRESHOLD_MULTIPLIER: float = 1.5

    # ── Strategy Learning (Vision AI) ──
    # Vision is DECOUPLED from the text LLM so a text-only model (llama-3.3-70b) doesn't force a broken
    # vision model. Default: groq's Llama-4 Scout (multimodal, verified vision-capable, no extra key).
    VISION_PROVIDER: Optional[str] = None   # None → fall back to LLM_PROVIDER
    VISION_MODEL: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    VISION_API_KEY: Optional[str] = None    # None → fall back to LLM_API_KEY
    VISION_MAX_TOKENS: int = 2000
    LEARNING_MIN_EXAMPLES: int = 5
    LEARNING_VECTOR_COLLECTION: str = "ptis_strategy_learning"
    SCREENSHOT_STORAGE_PATH: str = "./data/screenshots"
    SCREENSHOT_MAX_SIZE_MB: int = 10

    # ── Strategy Knowledge Engine (SKE) ──
    SKE_MIN_TRADES_DEVELOPING: int = 10
    SKE_MIN_TRADES_VALIDATED: int = 20
    SKE_MIN_TRADES_PROVEN: int = 50
    SKE_DQAL_MIN_PASS: float = 60.0
    SKE_VARIANT_MIN_CLUSTER_SIZE: int = 5
    SKE_SIGNAL_EXPIRY_MINUTES: int = 30
    SKE_VECTOR_COLLECTION: str = "ptis_ske_features"

    class Config:
        env_file = _ENV_FILE
        env_file_encoding = "utf-8"

    # ── Convenience properties ──

    @property
    def upstox_access_token(self) -> Optional[str]:
        return self.UPSTOX_ACCESS_TOKEN

    @property
    def upstox_api_key(self) -> Optional[str]:
        return self.UPSTOX_API_KEY

    def validate_config(self) -> List[str]:
        """
        Run startup sanity checks.
        In production (DEBUG=False), critical failures raise RuntimeError immediately.
        Returns list of non-fatal warning messages.
        """
        warnings: List[str] = []

        # ── JWT secret ──
        _dev_secret = "ptis-dev-secret-change-in-production-2026"
        if self.PTIS_JWT_SECRET == _dev_secret:
            if self.DEBUG:
                warnings.append(
                    "PTIS_JWT_SECRET is still the default dev value — safe only in DEBUG mode"
                )
            # Hard block in api/main.py lifespan (kept there for explicitness)

        if len(self.PTIS_JWT_SECRET) < 32:
            msg = (
                f"PTIS_JWT_SECRET is too short ({len(self.PTIS_JWT_SECRET)} chars). "
                "Minimum 32 characters required for HS256. "
                "Generate with: python -c \"import secrets; print(secrets.token_urlsafe(64))\""
            )
            if not self.DEBUG:
                raise RuntimeError(msg)
            warnings.append(msg)

        # ── CORS wildcard ──
        if "*" in self.CORS_ORIGINS:
            msg = "CORS_ORIGINS contains wildcard '*' — this allows any origin to call the API"
            if not self.DEBUG:
                raise RuntimeError(msg)
            warnings.append(msg)

        # ── Database URL format ──
        if not self.DATABASE_URL.startswith("postgresql://") and not self.DATABASE_URL.startswith("postgresql+asyncpg://"):
            warnings.append(f"DATABASE_URL does not start with postgresql:// — got: {self.DATABASE_URL[:30]}...")

        # ── LLM key ──
        if not self.LLM_API_KEY:
            warnings.append("LLM_API_KEY is not set — coaching and discovery features will not work")

        # ── Risk param bounds ──
        if self.DEFAULT_RISK_PERCENT > self.MAX_RISK_PERCENT:
            warnings.append(
                f"DEFAULT_RISK_PERCENT ({self.DEFAULT_RISK_PERCENT}) > MAX_RISK_PERCENT ({self.MAX_RISK_PERCENT})"
            )
        if self.MAX_RISK_PERCENT > 5.0:
            msg = f"MAX_RISK_PERCENT ({self.MAX_RISK_PERCENT}%) is dangerously high for real-money trading (>5%)"
            if not self.DEBUG:
                raise RuntimeError(msg)
            warnings.append(msg)
        if self.MAX_DAILY_LOSS_PERCENT > 10.0:
            msg = f"MAX_DAILY_LOSS_PERCENT ({self.MAX_DAILY_LOSS_PERCENT}%) exceeds 10% — refusing to start with this setting"
            raise RuntimeError(msg)  # Always fatal — no one should ever lose >10%/day

        # ── Sentry in production ──
        if not self.DEBUG and not self.SENTRY_DSN:
            warnings.append("Running in production (DEBUG=false) without SENTRY_DSN for error tracking")

        return warnings


settings = Settings()


def get_settings() -> Settings:
    """Factory function for settings access (alternative to singleton import)."""
    return settings
