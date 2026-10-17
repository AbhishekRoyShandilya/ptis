import pytest
from unittest.mock import MagicMock
from httpx import AsyncClient, ASGITransport
from api.main import app
from api.auth import get_current_user
from core.models import User


@pytest.fixture
async def auth_client():
    stub_user = MagicMock(spec=User)
    stub_user.id = "test-user-id"
    stub_user.trader_id = "test-trader"
    stub_user.is_active = True
    app.dependency_overrides[get_current_user] = lambda: stub_user
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            yield client
    finally:
        app.dependency_overrides.pop(get_current_user, None)


@pytest.mark.asyncio
async def test_cognitive_alerts_endpoint_returns_list(auth_client, monkeypatch):
    async def fake_alerts(self, trader_id: str):
        return [{"id": "1", "message": "test"}]

    from intelligence.realtime_service import RealtimeIntelligenceService
    monkeypatch.setattr(RealtimeIntelligenceService, "get_recent_cognitive_alerts", fake_alerts)

    response = await auth_client.get("/api/v1/intelligence/cognitive-alerts")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 1
    assert isinstance(data["alerts"], list)
