# config/logging_config.py
"""
Structured Logging Configuration
===================================
JSON logging for production, human-readable for development.

Production (LOG_FORMAT=json):
  {"timestamp": "2026-03-21T10:30:00Z", "level": "INFO", "logger": "core.trade_pipeline", "message": "Trade entered", ...}

Development (LOG_FORMAT=text, default):
  2026-03-21 10:30:00 | INFO     | core.trade_pipeline | Trade entered
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any

from config.settings import settings


class JSONFormatter(logging.Formatter):
    """Structured JSON log formatter for production."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Include extra fields if present
        for key in ("trade_id", "symbol", "user_id", "request_id", "duration_ms"):
            val = getattr(record, key, None)
            if val is not None:
                log_entry[key] = val

        return json.dumps(log_entry, default=str)


class TextFormatter(logging.Formatter):
    """Human-readable log formatter for development."""

    def __init__(self):
        super().__init__(
            fmt="%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


def setup_logging() -> None:
    """Configure application logging based on settings."""
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    log_format = os.getenv("LOG_FORMAT", "text").lower()

    # Choose formatter
    if log_format == "json":
        formatter = JSONFormatter()
    else:
        formatter = TextFormatter()

    # Console handler — force UTF-8 on Windows (avoids cp1252 UnicodeEncodeError)
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(log_level)
    console.setFormatter(formatter)

    # File handler with rotation (always JSON for machine parsing)
    os.makedirs("data", exist_ok=True)
    from logging.handlers import RotatingFileHandler
    file_handler = RotatingFileHandler(
        "data/ptis.log", maxBytes=50 * 1024 * 1024, backupCount=5,  # 50 MB, keep 5 backups
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(JSONFormatter())

    # Root logger
    root = logging.getLogger()
    # Clear any existing handlers to prevent duplicates on reload
    root.handlers.clear()
    root.setLevel(log_level)
    root.addHandler(console)
    root.addHandler(file_handler)

    # Quiet noisy libraries
    for lib in ("sqlalchemy.engine", "urllib3", "httpcore", "httpx", "chromadb", "sentence_transformers"):
        logging.getLogger(lib).setLevel(logging.WARNING)

    logging.info("Logging initialized: level=%s, format=%s", settings.LOG_LEVEL, log_format)
