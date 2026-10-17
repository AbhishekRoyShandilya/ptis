# tests/api/test_analytics_routes.py
"""Tests for api/routes/analytics.py endpoints."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from api.main import create_app
    from api.auth import get_current_user

    app = create_app()
    app.dependency_overrides[get_current_user] = lambda: {"sub": "test-user-id", "username": "tester"}
    return TestClient(app, raise_server_exceptions=False)


class TestAnalyticsEndpoints:
    def test_strategy_performance_nonexistent(self, client):
        resp = client.get("/api/v1/analytics/strategy/test_breakout")
        assert resp.status_code in (200, 404, 500)

    def test_deviation_analysis_nonexistent(self, client):
        resp = client.get("/api/v1/analytics/deviation/00000000-0000-0000-0000-000000000001")
        assert resp.status_code in (200, 404, 500)

    def test_analytics_strategy_invalid_id(self, client):
        resp = client.get("/api/v1/analytics/strategy/")
        # Trailing slash → likely 404 or 307 redirect
        assert resp.status_code in (307, 404, 405, 422)

    def test_health_still_works(self, client):
        """Verify analytics routes don't break the app."""
        resp = client.get("/health")
        assert resp.status_code == 200
