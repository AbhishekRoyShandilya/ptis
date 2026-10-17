# tests/conftest.py
"""Shared test fixtures."""

import pytest
import pytest_asyncio
from datetime import datetime, timezone
from unittest.mock import MagicMock
from sqlalchemy import DateTime, create_engine, event
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.attributes import set_committed_value

from core.models import Base


# The models use Postgres column types; the unit/integration tests run on in-memory SQLite. Render them as the
# nearest SQLite types (SQLAlchemy's Uuid type already stores non-native UUIDs as strings).
@compiles(UUID, "sqlite")
def _uuid_on_sqlite(type_, compiler, **kw):
    return "CHAR(32)"


@compiles(JSONB, "sqlite")
@compiles(ARRAY, "sqlite")
def _json_on_sqlite(type_, compiler, **kw):
    return "JSON"


# Postgres `timestamptz` columns come back tz-aware; SQLite drops the zone. Restore UTC on load/refresh so the code
# under test sees the same datetimes it sees in production.
def _restore_utc(target, *_args):
    for prop in sa_inspect(type(target)).column_attrs:
        col_type = prop.columns[0].type
        if isinstance(col_type, DateTime) and col_type.timezone:
            value = target.__dict__.get(prop.key)
            if isinstance(value, datetime) and value.tzinfo is None:
                set_committed_value(target, prop.key, value.replace(tzinfo=timezone.utc))


event.listen(Base, "load", _restore_utc, propagate=True)
event.listen(Base, "refresh", _restore_utc, propagate=True)


@pytest.fixture(scope="session")
def engine():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def db(engine):
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.rollback()
    session.close()


@pytest.fixture
async def async_client():
    """Async HTTP client for FastAPI endpoint tests.
    Overrides auth dependency so tests don't need real JWT tokens.
    All endpoint URLs must use /api/v1 prefix.
    """
    from httpx import AsyncClient, ASGITransport
    from api.main import app
    from api.auth import get_current_user
    from core.models import User

    # Stub user — satisfies get_current_user dependency
    stub_user = MagicMock(spec=User)
    stub_user.id = "test-user-id"
    stub_user.trader_id = "test-trader"
    stub_user.username = "testuser"
    stub_user.is_active = True

    app.dependency_overrides[get_current_user] = lambda: stub_user
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            yield client
    finally:
        app.dependency_overrides.pop(get_current_user, None)


@pytest.fixture
def registry(tmp_path):
    """Registry loaded with test strategy YAMLs."""
    import yaml
    from strategy_engine.strategy_registry import StrategyRegistry

    # Write test strategy YAML
    strategy_yaml = {
        "strategy": {
            "name": "Test Breakout",
            "id": "test_breakout",
            "version": 1,
            "description": "Test strategy",
            "author": "test",
            "active": True,
            "market_regime": {
                "preferred": ["trending_up"],
                "avoid": ["low_vol"],
                "neutral": ["range"],
            },
            "entry": {
                "direction": "long",
                "confirmations": [
                    {
                        "type": "price_breakout",
                        "params": {"reference": "opening_range", "min_extension_pct": 0.1},
                        "required": True,
                        "weight": 1.0,
                    },
                    {
                        "type": "volume_surge",
                        "params": {"min_relative_volume": 1.5},
                        "required": True,
                        "weight": 0.8,
                    },
                    {
                        "type": "ema_alignment",
                        "params": {"fast_period": 9, "slow_period": 20},
                        "required": False,
                        "weight": 0.3,
                    },
                ],
                "avoid_conditions": [
                    {
                        "type": "time_outside_window",
                        "params": {"after": "11:30"},
                        "description": "avoid midday",
                    },
                ],
                "time_window": {
                    "start": "09:35",
                    "end": "11:00",
                    "timezone": "Asia/Kolkata",
                },
            },
            "exit": {
                "stop_loss": {"type": "structure", "params": {"reference": "opening_range_opposite"}},
                "targets": [
                    {"r_multiple": 1.0, "size_pct": 50},
                    {"r_multiple": 2.0, "size_pct": 50},
                ],
                "trailing": {"enabled": False},
                "time_exit": {"enabled": False},
            },
            "risk": {
                "max_risk_pct": 1.0,
                "max_daily_trades": 3,
                "min_r_target": 1.5,
            },
            "invalidation": [],
            "execution": {"entry_type": "stop", "preferred_timeframe": "5m"},
        }
    }

    yaml_path = tmp_path / "test_breakout.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(strategy_yaml, f)

    reg = StrategyRegistry(str(tmp_path))
    reg.initialize()
    return reg