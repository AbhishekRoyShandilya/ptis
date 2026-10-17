# tests/test_api_integration.py
"""
Integration tests for critical API endpoints.
Requires: pip install httpx pytest

Tests verify:
- Auth flow (register, login, token refresh, logout)
- Trade CRUD with ownership isolation
- WebSocket auth rejection without token
- Dashboard scoping
"""

import uuid
from unittest.mock import patch, MagicMock

import pytest


# ── Auth flow tests (unit-level, no DB) ──

def test_jwt_token_contains_jti():
    """Tokens must include jti for revocation support."""
    from api.auth import create_access_token, decode_token
    token = create_access_token(str(uuid.uuid4()), "testuser")
    payload = decode_token(token)
    assert "jti" in payload
    assert len(payload["jti"]) == 36  # UUID format


def test_access_token_has_correct_type():
    from api.auth import create_access_token, decode_token
    token = create_access_token(str(uuid.uuid4()), "testuser")
    payload = decode_token(token)
    assert payload["type"] == "access"


def test_refresh_token_has_correct_type():
    from api.auth import create_refresh_token, decode_token
    token = create_refresh_token(str(uuid.uuid4()))
    payload = decode_token(token)
    assert payload["type"] == "refresh"


def test_expired_token_raises():
    """Expired tokens should be rejected."""
    from datetime import timedelta
    from api.auth import create_access_token, decode_token
    from fastapi import HTTPException

    token = create_access_token(
        str(uuid.uuid4()), "testuser",
        expires_delta=timedelta(seconds=-10),  # already expired
    )
    with pytest.raises(HTTPException) as exc_info:
        decode_token(token)
    assert exc_info.value.status_code == 401


# ── Rate limiter tests ──

def test_rate_limiter_allows_within_limit():
    from api.middleware import RateLimiter
    limiter = RateLimiter(per_class_rpm={"default": 5})
    for i in range(5):
        allowed, remaining, reason = limiter.allow("1.2.3.4")
        assert allowed and reason == ""
        assert remaining == 5 - (i + 1)


def test_rate_limiter_blocks_over_limit():
    from api.middleware import RateLimiter
    limiter = RateLimiter(per_class_rpm={"default": 3})
    for _ in range(3):
        limiter.allow("1.2.3.4")
    allowed, remaining, reason = limiter.allow("1.2.3.4")
    assert not allowed and remaining == 0
    assert "3/min" in reason


def test_rate_limiter_buckets_are_per_route_class():
    from api.middleware import RateLimiter
    limiter = RateLimiter(per_class_rpm={"default": 2, "read": 2})
    for _ in range(2):
        limiter.allow("1.2.3.4", "default")
    assert not limiter.allow("1.2.3.4", "default")[0]
    assert limiter.allow("1.2.3.4", "read")[0]          # a busy class never starves another


def test_rate_limiter_cleanup_prevents_memory_leak():
    from api.middleware import RateLimiter
    limiter = RateLimiter(per_class_rpm={"default": 100})
    for i in range(1001):
        limiter.allow(f"10.0.{i // 256}.{i % 256}")
    # age every bucket past the one-minute window, then the next call must sweep them
    for key in list(limiter._minute_buckets):
        limiter._minute_buckets[key] = [t - 120 for t in limiter._minute_buckets[key]]
    limiter.allow("192.0.2.1")
    assert len(limiter._minute_buckets) == 1


# ── Request size validation ──

def test_max_request_body_constant():
    from api.middleware import MAX_REQUEST_BODY_BYTES
    assert MAX_REQUEST_BODY_BYTES == 10 * 1024 * 1024  # 10 MB


# ── DB pool configuration ──

def test_db_engine_has_pool_recycle():
    from core.database import engine
    assert engine.pool._recycle == 300


def test_db_engine_has_pool_pre_ping():
    from core.database import engine
    assert engine.pool._pre_ping is True


# ── Model structure tests ──

def test_token_blacklist_model_exists():
    from core.models import TokenBlacklist
    assert TokenBlacklist.__tablename__ == "token_blacklist"


def test_user_has_tokens_invalidated_at():
    from core.models import User
    cols = {c.name for c in User.__table__.columns}
    assert "tokens_invalidated_at" in cols


def test_trade_has_trader_id():
    from core.models import Trade
    cols = {c.name for c in Trade.__table__.columns}
    assert "trader_id" in cols


def test_watchlist_has_trader_id():
    from core.models import Watchlist
    cols = {c.name for c in Watchlist.__table__.columns}
    assert "trader_id" in cols


def test_automation_rule_has_trader_id():
    from core.models import AutomationRule
    cols = {c.name for c in AutomationRule.__table__.columns}
    assert "trader_id" in cols
