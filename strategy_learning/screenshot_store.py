# strategy_learning/screenshot_store.py
"""
Screenshot storage with magic-byte validation, deduplication, and cleanup.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
from pathlib import Path
from typing import Optional
from uuid import UUID

from config.settings import settings

logger = logging.getLogger(__name__)

# Extension → expected magic-byte prefixes
_MAGIC_BYTES: dict[str, list[bytes]] = {
    ".png": [b"\x89PNG"],
    ".jpg": [b"\xff\xd8\xff"],
    ".jpeg": [b"\xff\xd8\xff"],
    ".webp": [b"RIFF"],
}

ALLOWED_EXTENSIONS = set(_MAGIC_BYTES.keys())


class ScreenshotStore:
    """Manages chart-screenshot storage on the local filesystem."""

    def __init__(self, base_path: str | None = None):
        self.base_path = Path(base_path or settings.SCREENSHOT_STORAGE_PATH)
        self.base_path.mkdir(parents=True, exist_ok=True)

    # ── public API ────────────────────────────────────────────

    def save(
        self,
        profile_id: UUID,
        session_id: UUID,
        file_bytes: bytes,
        filename: str,
    ) -> str:
        """
        Validate, deduplicate, and persist a screenshot.

        Returns
        -------
        str
            The relative storage path (use for DB storage).
        """
        ext = self._validate(file_bytes, filename)

        directory = self._session_dir(profile_id, session_id)
        directory.mkdir(parents=True, exist_ok=True)

        file_hash = self.compute_hash(file_bytes)
        safe_name = f"{file_hash[:16]}{ext}"
        file_path = directory / safe_name

        # Deduplicate — if identical file already exists, skip write
        if file_path.exists():
            logger.debug("Duplicate screenshot skipped: %s", file_path)
            return str(file_path.relative_to(self.base_path))

        file_path.write_bytes(file_bytes)
        logger.info(
            "Saved screenshot: %s (%d bytes, hash=%s)",
            file_path, len(file_bytes), file_hash[:12],
        )
        return str(file_path.relative_to(self.base_path))

    def compute_hash(self, file_bytes: bytes) -> str:
        return hashlib.sha256(file_bytes).hexdigest()

    def resolve(self, relative_path: str) -> Path:
        """Resolve a stored relative path to an absolute Path."""
        return self.base_path / relative_path

    def exists(self, relative_path: str) -> bool:
        return self.resolve(relative_path).is_file()

    def read(self, relative_path: str) -> bytes:
        """Read screenshot bytes.  Raises FileNotFoundError."""
        path = self.resolve(relative_path)
        if not path.is_file():
            raise FileNotFoundError(f"Screenshot not found: {path}")
        return path.read_bytes()

    def delete_session_screenshots(
        self, profile_id: UUID, session_id: UUID
    ) -> int:
        """Remove all files for a session.  Returns count deleted."""
        directory = self._session_dir(profile_id, session_id)
        if not directory.exists():
            return 0

        count = sum(1 for f in directory.iterdir() if f.is_file())
        shutil.rmtree(directory)
        logger.info("Deleted %d screenshots for session %s", count, session_id)
        return count

    def delete_profile_screenshots(self, profile_id: UUID) -> int:
        """Remove *all* screenshots for a profile."""
        directory = self.base_path / str(profile_id)
        if not directory.exists():
            return 0
        count = sum(1 for f in directory.rglob("*") if f.is_file())
        shutil.rmtree(directory)
        logger.info("Deleted %d screenshots for profile %s", count, profile_id)
        return count

    def disk_usage_bytes(self, profile_id: Optional[UUID] = None) -> int:
        """Total bytes consumed (optionally scoped to a profile)."""
        root = (self.base_path / str(profile_id)) if profile_id else self.base_path
        if not root.exists():
            return 0
        return sum(f.stat().st_size for f in root.rglob("*") if f.is_file())

    # ── internals ─────────────────────────────────────────────

    def _session_dir(self, profile_id: UUID, session_id: UUID) -> Path:
        return self.base_path / str(profile_id) / str(session_id)

    def _validate(self, file_bytes: bytes, filename: str) -> str:
        """
        Validate size, extension, and magic bytes.

        Returns the normalised extension.
        """
        # Size check
        max_bytes = settings.SCREENSHOT_MAX_SIZE_MB * 1024 * 1024
        if len(file_bytes) > max_bytes:
            raise ValueError(
                f"Screenshot too large: {len(file_bytes):,} bytes "
                f"(max {settings.SCREENSHOT_MAX_SIZE_MB} MB)"
            )
        if len(file_bytes) < 64:
            raise ValueError("File too small to be a valid image")

        # Extension check
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise ValueError(
                f"Unsupported format '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            )

        # Magic-byte check
        prefixes = _MAGIC_BYTES[ext]
        if not any(file_bytes[:len(p)] == p for p in prefixes):
            raise ValueError(
                f"File content does not match '{ext}' format "
                "(magic-byte mismatch — possible corruption or mislabelled file)"
            )

        return ext