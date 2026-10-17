# api/middleware.py
"""
API Middleware
===============
Request logging, rate limiting, and performance tracking.

Applied in api/main.py via app.add_middleware(RequestLoggingMiddleware).
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

SLOW_REQUEST_THRESHOLD_MS = 2000  # Log WARNING for requests slower than 2s
MAX_REQUEST_BODY_BYTES = 10 * 1024 * 1024  # 10 MB max request body


# ── Rate-limit configuration ────────────────────────────────────────
# Loaded from env at import time. Single trader self-hosted: localhost is
# whitelisted by default. For production multi-tenant: set RATE_LIMIT_EXEMPT_IPS=""
# and rely on per-class limits below.

def _parse_csv_env(name: str, default: str) -> List[str]:
    raw = os.getenv(name, default)
    return [s.strip() for s in raw.split(",") if s.strip()]


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


# IPs that bypass rate limiting entirely (the trader's own machine)
RATE_LIMIT_EXEMPT_IPS = set(_parse_csv_env(
    "RATE_LIMIT_EXEMPT_IPS",
    "127.0.0.1,::1,localhost",
))

# Per-route-class limits per minute. Hour limits are 10x minute limits.
# Defaults sized for an active trader's dashboard (page load + polling).
RATE_LIMIT_PER_CLASS = {
    "auth":      _int_env("RATE_LIMIT_AUTH_RPM",      30),    # login/logout — strict
    "expensive": _int_env("RATE_LIMIT_EXPENSIVE_RPM", 30),    # LLM, backtest, smart-seed
    "default":   _int_env("RATE_LIMIT_DEFAULT_RPM",   180),   # most endpoints
    "read":      _int_env("RATE_LIMIT_READ_RPM",      300),   # high-frequency polling (watchlist, alerts, ltp)
}

# Path → class classification. Longest-prefix match wins; "default" fallback.
# Expensive routes invoke LLMs, broker APIs, or long-running computations.
_EXPENSIVE_PREFIXES = (
    "/api/v1/chat/co-plan",
    "/api/v1/chat/discretionary",
    "/api/v1/intelligence/evaluate",
    "/api/v1/discovery/discover",
    "/api/v1/research/discover-continuous",
    "/api/v1/research/run",
    "/api/v1/avt/start",
    "/api/v1/watchlist/smart-seed",
    "/api/v1/bootstrap/run",
    "/api/v1/bootstrap/finalize",
    "/api/v1/learning/teach",
)

# Read routes are high-frequency polls — they should NOT be rate-limited as tightly.
_READ_PREFIXES = (
    "/api/v1/watchlist",
    "/api/v1/intelligence/cognitive-alerts",
    "/api/v1/regime/current",
    "/api/v1/dashboard",
    "/api/v1/behavior/discipline",
    "/api/v1/behavior/deviations",
    "/api/v1/profile",
    "/api/v1/trades",
    "/api/v1/strategies",
    "/api/v1/journal",
    "/api/v1/avt/status",
    "/api/v1/research/findings",
    "/api/v1/research/tasks",
    "/api/v1/research/drafts",
    "/api/v1/bootstrap/patterns",
)

_AUTH_PREFIXES = (
    "/api/v1/auth/login",
    "/api/v1/auth/register",
    "/api/v1/auth/logout",
)


def classify_route(path: str) -> str:
    """Return 'auth' | 'expensive' | 'read' | 'default' for a request path."""
    if any(path.startswith(p) for p in _AUTH_PREFIXES):
        return "auth"
    if any(path.startswith(p) for p in _EXPENSIVE_PREFIXES):
        return "expensive"
    if any(path.startswith(p) for p in _READ_PREFIXES):
        return "read"
    return "default"


# ── Rate limiter (in-memory, per-IP × per-class) ──
# TODO(prod): Replace with Redis-backed rate limiting for multi-replica deployments.
# In-memory limits are per-process, so multiple replicas make limits inconsistent.

class RateLimiter:
    """
    Per-IP, per-route-class rate limiter.

    Each class (auth/expensive/default/read) has its own bucket so a burst
    of cheap reads doesn't starve expensive operations or vice versa.

    IPs in RATE_LIMIT_EXEMPT_IPS bypass entirely — useful for single-trader
    self-hosted setups where 127.0.0.1 is the only client.
    """

    def __init__(self, per_class_rpm: Optional[Dict[str, int]] = None):
        self._per_class_rpm = per_class_rpm or RATE_LIMIT_PER_CLASS
        # buckets keyed by (client_ip, route_class)
        self._minute_buckets: Dict[Tuple[str, str], list] = defaultdict(list)
        self._hour_buckets: Dict[Tuple[str, str], list] = defaultdict(list)

    def allow(self, client_ip: str, route_class: str = "default") -> Tuple[bool, int, str]:
        """Returns (allowed, remaining_per_minute, reason)."""
        # Whitelist exempt IPs (own trader's machine in self-hosted mode)
        if client_ip in RATE_LIMIT_EXEMPT_IPS:
            return True, 9999, ""

        now = time.time()

        # Periodic cleanup
        if len(self._minute_buckets) > 1000:
            self._cleanup(now)

        rpm = self._per_class_rpm.get(route_class, self._per_class_rpm["default"])
        rph = rpm * 10  # hour cap = 10x minute cap

        key = (client_ip, route_class)
        minute_cutoff = now - 60
        hour_cutoff = now - 3600

        self._minute_buckets[key] = [
            t for t in self._minute_buckets[key] if t > minute_cutoff
        ]
        self._hour_buckets[key] = [
            t for t in self._hour_buckets[key] if t > hour_cutoff
        ]

        if len(self._minute_buckets[key]) >= rpm:
            return False, 0, f"Rate limit exceeded: {rpm}/min for {route_class} routes"
        if len(self._hour_buckets[key]) >= rph:
            return False, 0, f"Rate limit exceeded: {rph}/hour for {route_class} routes"

        self._minute_buckets[key].append(now)
        self._hour_buckets[key].append(now)
        remaining = rpm - len(self._minute_buckets[key])
        return True, remaining, ""

    def get_limit(self, route_class: str) -> int:
        return self._per_class_rpm.get(route_class, self._per_class_rpm["default"])

    # Backward-compat property used by callers that read `._rpm` for headers
    @property
    def _rpm(self) -> int:
        return self._per_class_rpm.get("default", 180)

    def _cleanup(self, now: float) -> None:
        minute_cutoff = now - 60
        stale = [k for k, ts in self._minute_buckets.items() if not ts or ts[-1] < minute_cutoff]
        for k in stale:
            self._minute_buckets.pop(k, None)
            self._hour_buckets.pop(k, None)


# Module-level singletons
_rate_limiter = RateLimiter()

# Log effective config at startup so the operator can verify
logger.info(
    "Rate limit config: exempt_ips=%s, limits/min=%s",
    sorted(RATE_LIMIT_EXEMPT_IPS), RATE_LIMIT_PER_CLASS,
)


class RequestMetrics:
    """In-memory per-endpoint request count and latency tracking."""

    def __init__(self):
        self._counts: Dict[str, int] = defaultdict(int)
        self._total_ms: Dict[str, float] = defaultdict(float)
        self._errors: Dict[str, int] = defaultdict(int)

    def record(self, method: str, path: str, status: int, duration_ms: float):
        key = f"{method} {path}"
        self._counts[key] += 1
        self._total_ms[key] += duration_ms
        if status >= 500:
            self._errors[key] += 1

    def summary(self) -> Dict:
        """Return top-level metrics for /metrics endpoint."""
        total_requests = sum(self._counts.values())
        total_errors = sum(self._errors.values())
        endpoints = []
        for key in sorted(self._counts, key=lambda k: self._counts[k], reverse=True)[:20]:
            count = self._counts[key]
            avg_ms = self._total_ms[key] / count if count else 0
            endpoints.append({
                "endpoint": key,
                "requests": count,
                "avg_latency_ms": round(avg_ms, 1),
                "errors": self._errors.get(key, 0),
            })
        return {
            "total_requests": total_requests,
            "total_errors": total_errors,
            "top_endpoints": endpoints,
        }


_metrics = RequestMetrics()


def get_request_metrics() -> RequestMetrics:
    return _metrics


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Logs every request with timing info, applies rate limiting,
    tracks per-endpoint metrics, and adds X-Request-ID for tracing.
    """

    SKIP_PATHS = {"/health", "/docs", "/openapi.json", "/redoc", "/metrics"}

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        path = request.url.path

        # Assign request ID for tracing
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())[:12]

        # Skip logging for health/docs/metrics
        if path in self.SKIP_PATHS:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response

        # Skip WebSocket
        if "websocket" in path.lower():
            return await call_next(request)

        # Request size limit
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_REQUEST_BODY_BYTES:
            return JSONResponse(
                status_code=413,
                content={"detail": f"Request body too large (max {MAX_REQUEST_BODY_BYTES // (1024*1024)} MB)"},
                headers={"X-Request-ID": request_id},
            )

        # Rate limiting (per-IP × per-route-class, with localhost exemption)
        client_ip = request.client.host if request.client else "unknown"
        route_class = classify_route(path)
        allowed, remaining, reason = _rate_limiter.allow(client_ip, route_class)
        if not allowed:
            logger.warning(
                "Rate limited: %s %s [%s class] from %s — %s [rid=%s]",
                request.method, path, route_class, client_ip, reason, request_id,
            )
            return JSONResponse(
                status_code=429,
                content={"detail": reason, "route_class": route_class},
                headers={
                    "Retry-After": "60",
                    "X-RateLimit-Limit": str(_rate_limiter.get_limit(route_class)),
                    "X-RateLimit-Class": route_class,
                    "X-RateLimit-Remaining": "0",
                    "X-Request-ID": request_id,
                },
            )

        # Timing
        start = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.error(
                "ERROR %s %s [%.1fms] %s: %s [rid=%s]",
                request.method, path, duration_ms, type(e).__name__, e, request_id,
            )
            _metrics.record(request.method, path, 500, duration_ms)
            raise

        duration_ms = (time.perf_counter() - start) * 1000
        status = response.status_code

        # Record metrics
        _metrics.record(request.method, path, status, duration_ms)

        # Log level based on status
        level = logging.INFO if status < 400 else logging.WARNING if status < 500 else logging.ERROR

        # Slow request warning
        if duration_ms > SLOW_REQUEST_THRESHOLD_MS:
            logger.warning(
                "SLOW %s %s %d [%.1fms] client=%s [rid=%s]",
                request.method, path, status, duration_ms, client_ip, request_id,
            )
        else:
            logger.log(
                level,
                "%s %s %d [%.1fms] client=%s [rid=%s]",
                request.method, path, status, duration_ms, client_ip, request_id,
            )

        # Add response headers
        response.headers["X-Process-Time-Ms"] = f"{duration_ms:.1f}"
        response.headers["X-RateLimit-Limit"] = str(_rate_limiter.get_limit(route_class))
        response.headers["X-RateLimit-Class"] = route_class
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-Request-ID"] = request_id

        # Security headers (applied to all API responses)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        # HSTS only over HTTPS (not local dev)
        try:
            from config.settings import settings as _settings
            if not _settings.DEBUG:
                response.headers["Strict-Transport-Security"] = (
                    "max-age=31536000; includeSubDomains"
                )
        except Exception:
            pass

        return response
