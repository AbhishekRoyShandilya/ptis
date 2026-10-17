# api/schemas/common_schemas.py
"""
Common response schemas used across multiple endpoints.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class StatusResponse(BaseModel):
    """Generic status response."""
    status: str


class PaginatedParams(BaseModel):
    """Common pagination parameters."""
    limit: int = 50
    offset: int = 0


class ErrorResponse(BaseModel):
    """Standard error response."""
    detail: str
    type: Optional[str] = None
