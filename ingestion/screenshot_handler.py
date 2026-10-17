# ingestion/screenshot_handler.py
"""Screenshot validation and preprocessing utilities."""

from __future__ import annotations

import io
import logging
from pathlib import Path

from config.settings import settings

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
ALLOWED_MIMES = {"image/png", "image/jpeg", "image/webp"}


def validate_screenshot(file_bytes: bytes, filename: str) -> bool:
    """Validate screenshot file size and format."""
    max_bytes = settings.SCREENSHOT_MAX_SIZE_MB * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise ValueError(
            f"File too large: {len(file_bytes) / (1024*1024):.1f} MB "
            f"(max {settings.SCREENSHOT_MAX_SIZE_MB} MB)"
        )

    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported format: {ext}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    return True


def get_media_type(filename: str) -> str:
    """Get MIME type from filename extension."""
    ext = Path(filename).suffix.lower()
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }
    return mime_map.get(ext, "image/png")


def resize_if_needed(file_bytes: bytes, max_dim: int = 2048) -> bytes:
    """Resize image if dimensions exceed max_dim. Requires Pillow."""
    try:
        from PIL import Image
    except ImportError:
        logger.debug("Pillow not installed, skipping resize check")
        return file_bytes

    img = Image.open(io.BytesIO(file_bytes))
    w, h = img.size

    if w <= max_dim and h <= max_dim:
        return file_bytes

    # Scale down preserving aspect ratio
    ratio = min(max_dim / w, max_dim / h)
    new_w = int(w * ratio)
    new_h = int(h * ratio)

    img = img.resize((new_w, new_h), Image.LANCZOS)

    buf = io.BytesIO()
    fmt = img.format or "PNG"
    img.save(buf, format=fmt)
    resized = buf.getvalue()

    logger.info("Resized screenshot from %dx%d to %dx%d", w, h, new_w, new_h)
    return resized
