# core/database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
from config.settings import settings
from core.models import Base

# SQL echo is DECOUPLED from DEBUG: echoing every query floods the terminal (hiding AVT scan/trade logs) and
# adds real overhead under frequent polling. Off by default; set DB_ECHO=true only when debugging SQL.
_DB_ECHO = getattr(settings, "DB_ECHO", False)

engine = create_engine(
    settings.DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_recycle=300,
    pool_pre_ping=True,
    echo=_DB_ECHO,
)

# Async session factory (used by chat/WebSocket routes)
try:
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker as _sessionmaker

    _async_url = settings.DATABASE_URL.replace(
        "postgresql://", "postgresql+asyncpg://", 1
    ).replace(
        "postgres://", "postgresql+asyncpg://", 1
    )
    _async_engine = create_async_engine(_async_url, echo=_DB_ECHO)
    async_session_factory = _sessionmaker(
        _async_engine, class_=AsyncSession, expire_on_commit=False
    )
except Exception as _async_err:
    import logging as _logging
    _logging.getLogger(__name__).warning(
        "Async DB not available (install asyncpg): %s — WebSocket routes will use sync fallback", _async_err
    )
    async_session_factory = None

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def init_db():
    """Create all tables."""
    Base.metadata.create_all(bind=engine)


@contextmanager
def get_db():
    """Database session context manager."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_db_dependency():
    """FastAPI dependency."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()