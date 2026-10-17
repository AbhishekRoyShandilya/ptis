# ingestion/token_store.py
"""
Persistent token store for broker OAuth tokens.
Saves to data/broker_tokens.json so tokens survive server restarts.
Falls back to .env values if no saved token exists.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _token_file() -> Path:
    """Return token file path — respects PTIS_ENV so test and prod never share tokens."""
    try:
        from config.settings import settings
        return Path(settings.TOKEN_STORE_PATH)
    except Exception:
        return Path(__file__).parent.parent / "data" / "broker_tokens.json"


def _load() -> dict:
    f = _token_file()
    if f.exists():
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Could not read %s: %s", f, e)
    return {}


def _save(data: dict) -> None:
    f = _token_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(data, indent=2), encoding="utf-8")


def get_upstox_token() -> Optional[str]:
    """
    Return the best available Upstox access token.
    Priority: saved token file → UPSTOX_ACCESS_TOKEN env var.
    """
    stored = _load().get("upstox_access_token")
    if stored:
        return stored

    from config.settings import settings
    return settings.UPSTOX_ACCESS_TOKEN or None


def save_upstox_token(token: str) -> None:
    """Persist a new Upstox access token."""
    data = _load()
    data["upstox_access_token"] = token
    data["upstox_updated_at"] = datetime.now(timezone.utc).isoformat()
    _save(data)
    logger.info("Upstox access token saved to token store")
