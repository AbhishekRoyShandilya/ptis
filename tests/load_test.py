# tests/load_test.py
"""
Simple load test for PTIS API endpoints.

Usage:
    pip install httpx
    python tests/load_test.py

Not a full k6/locust replacement — just a quick validation that
critical endpoints respond under concurrent load.
"""

import asyncio
import time
import sys
from dataclasses import dataclass, field

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore
    if __name__ == "__main__":
        print("Install httpx: pip install httpx")
        sys.exit(1)


BASE_URL = "http://localhost:8000"
CONCURRENT_USERS = 10
REQUESTS_PER_USER = 20


@dataclass
class LoadTestResult:
    endpoint: str
    total_requests: int = 0
    success_count: int = 0
    error_count: int = 0
    total_ms: float = 0.0
    min_ms: float = float("inf")
    max_ms: float = 0.0
    status_codes: dict = field(default_factory=dict)

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.total_requests if self.total_requests else 0

    @property
    def success_rate(self) -> float:
        return self.success_count / self.total_requests * 100 if self.total_requests else 0


async def login(client: httpx.AsyncClient) -> str:
    """Login and return access token."""
    resp = await client.post(f"{BASE_URL}/api/v1/auth/login", json={
        "username": "admin",
        "password": "admin123",
    })
    if resp.status_code == 200:
        return resp.json()["access_token"]
    return ""


async def hit_endpoint(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    token: str,
    result: LoadTestResult,
):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    start = time.perf_counter()
    try:
        if method == "GET":
            resp = await client.get(url, headers=headers)
        else:
            resp = await client.post(url, headers=headers, json={})
        elapsed = (time.perf_counter() - start) * 1000

        result.total_requests += 1
        result.total_ms += elapsed
        result.min_ms = min(result.min_ms, elapsed)
        result.max_ms = max(result.max_ms, elapsed)

        code = resp.status_code
        result.status_codes[code] = result.status_codes.get(code, 0) + 1
        if code < 400:
            result.success_count += 1
        else:
            result.error_count += 1
    except Exception:
        result.total_requests += 1
        result.error_count += 1


async def run_user(
    client: httpx.AsyncClient,
    token: str,
    endpoints: list,
    results: dict,
    requests_per: int,
):
    for _ in range(requests_per):
        for method, path in endpoints:
            await hit_endpoint(client, method, f"{BASE_URL}{path}", token, results[path])


async def main():
    endpoints = [
        ("GET", "/health"),
        ("GET", "/api/v1/dashboard/"),
        ("GET", "/api/v1/trades"),
        ("GET", "/api/v1/automation/guardrails/status"),
    ]

    results = {path: LoadTestResult(endpoint=path) for _, path in endpoints}

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Login
        token = await login(client)
        if not token:
            print("Login failed — make sure the server is running and 'admin' user exists")
            print("Continuing with unauthenticated endpoints only...")

        print(f"\nRunning load test: {CONCURRENT_USERS} users x {REQUESTS_PER_USER} requests")
        print(f"Endpoints: {len(endpoints)}")
        print("-" * 70)

        start = time.perf_counter()
        tasks = [
            run_user(client, token, endpoints, results, REQUESTS_PER_USER)
            for _ in range(CONCURRENT_USERS)
        ]
        await asyncio.gather(*tasks)
        total_time = time.perf_counter() - start

    print(f"\nCompleted in {total_time:.1f}s")
    print(f"{'Endpoint':<45} {'Reqs':>5} {'OK%':>6} {'Avg':>8} {'Min':>8} {'Max':>8}")
    print("-" * 88)

    for path, r in results.items():
        if r.min_ms == float("inf"):
            r.min_ms = 0
        print(
            f"{path:<45} {r.total_requests:>5} {r.success_rate:>5.1f}% "
            f"{r.avg_ms:>7.1f}ms {r.min_ms:>7.1f}ms {r.max_ms:>7.1f}ms"
        )
        if r.status_codes:
            codes = ", ".join(f"{k}:{v}" for k, v in sorted(r.status_codes.items()))
            print(f"  Status codes: {codes}")

    total_reqs = sum(r.total_requests for r in results.values())
    total_errors = sum(r.error_count for r in results.values())
    rps = total_reqs / total_time if total_time else 0
    print(f"\nTotal: {total_reqs} requests, {total_errors} errors, {rps:.0f} req/s")


if __name__ == "__main__":
    asyncio.run(main())
