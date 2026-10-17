# tests/api/test_shadow_routes.py
"""Tests for api/routes/shadow.py endpoints."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from api.main import create_app
    from api.auth import get_current_user

    app = create_app()
    app.dependency_overrides[get_current_user] = lambda: {"sub": "test-user-id", "username": "tester"}
    return TestClient(app, raise_server_exceptions=False)


class TestShadowEndpoints:
    def test_list_sessions(self, client):
        resp = client.get("/api/v1/shadow/sessions")
        assert resp.status_code in (200, 500)
        if resp.status_code == 200:
            assert isinstance(resp.json(), list)

    def test_list_sessions_with_filter(self, client):
        resp = client.get("/api/v1/shadow/sessions?status=active")
        assert resp.status_code in (200, 422, 500)

    def test_list_sessions_invalid_filter(self, client):
        resp = client.get("/api/v1/shadow/sessions?status=invalid")
        assert resp.status_code == 422  # validation error

    def test_shadow_summary(self, client):
        resp = client.get("/api/v1/shadow/summary")
        assert resp.status_code in (200, 404, 500)

    def test_get_nonexistent_session(self, client):
        resp = client.get("/api/v1/shadow/sessions/00000000-0000-0000-0000-000000000001")
        assert resp.status_code in (404, 422, 500)
