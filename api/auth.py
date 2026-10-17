# api/auth.py
"""
Authentication System
======================
JWT-based authentication for all PTIS API endpoints.

Architecture:
- Password hashing: bcrypt via passlib
- Token type: JWT (PyJWT) with HS256 signing
- Token lifetime: 24 hours (configurable)
- Refresh tokens: 7 days

Endpoints provided:
- POST /api/v1/auth/register — create account
- POST /api/v1/auth/login — get access + refresh tokens
- POST /api/v1/auth/refresh — get new access token
- GET  /api/v1/auth/me — current user info

Usage in routes:
    from api.auth import get_current_user
    @router.get("/protected")
    def protected_route(user = Depends(get_current_user)):
        ...
"""

from __future__ import annotations

import logging
import re
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import jwt
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from passlib.context import CryptContext
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session as DBSession

from config.settings import settings

logger = logging.getLogger(__name__)


# ── Auth-specific rate limiter (tighter than global) ──
class AuthRateLimiter:
    """Sliding-window rate limiter for login/register: 5 attempts per IP per minute."""

    def __init__(self, max_attempts: int = 5, window_seconds: int = 60):
        self._max = max_attempts
        self._window = window_seconds
        self._attempts: Dict[str, list] = defaultdict(list)

    def check(self, client_ip: str) -> bool:
        """Returns True if allowed, False if rate-limited."""
        import time as _time
        now = _time.time()
        cutoff = now - self._window
        self._attempts[client_ip] = [t for t in self._attempts[client_ip] if t > cutoff]
        if len(self._attempts[client_ip]) >= self._max:
            return False
        self._attempts[client_ip].append(now)
        return True


_auth_limiter = AuthRateLimiter()


def _check_auth_rate_limit(request: Request):
    """FastAPI dependency: enforce auth rate limiting."""
    client_ip = request.client.host if request.client else "unknown"
    if not _auth_limiter.check(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Too many authentication attempts. Try again in 60 seconds.",
            headers={"Retry-After": "60"},
        )

# ── Config ──
SECRET_KEY = settings.PTIS_JWT_SECRET
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto", bcrypt__rounds=12)

# Bearer token extractor
security = HTTPBearer(auto_error=False)


# ═══════════════════════════════════════════
#  Pydantic Models
# ═══════════════════════════════════════════

class UserCreate(BaseModel):
    """Registration request."""
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., min_length=5)
    password: str = Field(..., min_length=8)
    full_name: str = ""

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        if not re.match(pattern, v):
            raise ValueError("Invalid email format")
        return v.lower().strip()


class UserLogin(BaseModel):
    """Login request."""
    username: str
    password: str


class TokenResponse(BaseModel):
    """Token response."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds


class TokenRefresh(BaseModel):
    """Refresh request."""
    refresh_token: str


class UserResponse(BaseModel):
    """User info response."""
    user_id: str
    username: str
    email: str
    full_name: str
    created_at: str
    is_active: bool
    onboarding_completed: bool = False


class OnboardingUpdate(BaseModel):
    """Onboarding data persistence request."""
    onboarding_data: Dict[str, Any] = Field(default_factory=dict)
    onboarding_completed: bool = True


# ═══════════════════════════════════════════
#  User Store (DB-backed via SQLAlchemy)
# ═══════════════════════════════════════════

from core.models import User, TokenBlacklist
from core.database import get_db_dependency


class UserStore:
    """
    Database-backed user store using SQLAlchemy.
    Requires a DB session to be injected at construction time.
    """

    def __init__(self, db: DBSession):
        self._db = db

    def create_user(
        self,
        username: str,
        email: str,
        password: str,
        full_name: str = "",
    ) -> User:
        """Create a new user. Raises ValueError on duplicate username/email."""
        existing = self._db.query(User).filter(User.username == username).first()
        if existing:
            raise ValueError(f"Username '{username}' already exists")

        existing = self._db.query(User).filter(User.email == email).first()
        if existing:
            raise ValueError(f"Email '{email}' already registered")

        hashed = pwd_context.hash(password)

        user = User(
            username=username,
            email=email,
            hashed_password=hashed,
            full_name=full_name,
        )

        self._db.add(user)
        self._db.commit()
        self._db.refresh(user)

        logger.info("User created: %s (%s)", username, user.id)
        return user

    def get_by_username(self, username: str) -> Optional[User]:
        return self._db.query(User).filter(User.username == username).first()

    def get_by_id(self, user_id: str) -> Optional[User]:
        try:
            uid = uuid.UUID(user_id)
        except (ValueError, AttributeError):
            return None
        return self._db.query(User).filter(User.id == uid).first()

    def verify_password(self, plain: str, hashed: str) -> bool:
        return pwd_context.verify(plain, hashed)


# ═══════════════════════════════════════════
#  JWT Token Management
# ═══════════════════════════════════════════

def create_access_token(
    user_id: str,
    username: str,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """Create a JWT access token."""
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    )
    payload = {
        "sub": user_id,
        "username": username,
        "type": "access",
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "jti": str(uuid.uuid4()),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def create_refresh_token(user_id: str) -> str:
    """Create a JWT refresh token (longer-lived)."""
    expire = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    payload = {
        "sub": user_id,
        "type": "refresh",
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "jti": str(uuid.uuid4()),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> Dict[str, Any]:
    """Decode and validate a JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )


# ═══════════════════════════════════════════
#  FastAPI Dependencies
# ═══════════════════════════════════════════

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: DBSession = Depends(get_db_dependency),
) -> User:
    """
    FastAPI dependency: extract and validate the current user from JWT.

    Usage:
        @router.get("/protected")
        async def my_route(user = Depends(get_current_user)):
            ...
    """
    if settings.DEV_AUTH_BYPASS:
        # Return a real DB-backed dev user so all writes (trades, strategy
        # training, co-plan sessions) have a valid foreign key to attach to.
        # Creates the user on first run; subsequent calls just fetch it.
        # Never reachable in production (DEV_AUTH_BYPASS defaults to False).
        store = UserStore(db)
        dev_user = store.get_by_username("dev")
        if not dev_user:
            try:
                dev_user = store.create_user(
                    username="dev",
                    email="dev@localhost",
                    password="devpassword123",
                    full_name="Dev User",
                )
                logger.info("DEV_AUTH_BYPASS: created dev user in DB (id=%s)", dev_user.id)
            except ValueError:
                # Race condition — another request created it first
                dev_user = store.get_by_username("dev")
        return dev_user

    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_token(credentials.credentials)

    if payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type (expected access token)",
        )

    # Check token revocation
    jti = payload.get("jti")
    if jti:
        revoked = db.query(TokenBlacklist).filter(TokenBlacklist.jti == jti).first()
        if revoked:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has been revoked",
            )

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
        )

    store = UserStore(db)
    user = store.get_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled",
        )

    # Check logout-all: reject tokens issued before invalidation timestamp
    iat = payload.get("iat")
    if iat and user.tokens_invalidated_at:
        token_issued = datetime.fromtimestamp(iat, tz=timezone.utc)
        if token_issued < user.tokens_invalidated_at:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has been revoked (all sessions logged out)",
            )

    return user


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: DBSession = Depends(get_db_dependency),
) -> Optional[User]:
    """
    Like get_current_user but returns None instead of raising 401.
    For endpoints that work both authenticated and anonymous.
    """
    if not credentials:
        return None
    try:
        return await get_current_user(credentials, db)
    except HTTPException:
        return None


# ═══════════════════════════════════════════
#  Auth Router (mounted in api/main.py)
# ═══════════════════════════════════════════

auth_router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


@auth_router.post("/register", response_model=UserResponse)
async def register(
    data: UserCreate,
    request: Request,
    db: DBSession = Depends(get_db_dependency),
    _limit=Depends(_check_auth_rate_limit),
):
    """Register a new user account."""
    store = UserStore(db)
    try:
        user = store.create_user(
            username=data.username,
            email=data.email,
            password=data.password,
            full_name=data.full_name,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return UserResponse(
        user_id=str(user.id),
        username=user.username,
        email=user.email,
        full_name=user.full_name or "",
        created_at=user.created_at.isoformat(),
        is_active=user.is_active,
        onboarding_completed=getattr(user, "onboarding_completed", False) or False,
    )


@auth_router.post("/login", response_model=TokenResponse)
async def login(
    data: UserLogin,
    request: Request,
    db: DBSession = Depends(get_db_dependency),
    _limit=Depends(_check_auth_rate_limit),
):
    """Authenticate and get access + refresh tokens."""
    store = UserStore(db)
    user = store.get_by_username(data.username)

    if not user or not store.verify_password(data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled",
        )

    access = create_access_token(str(user.id), user.username)
    refresh = create_refresh_token(str(user.id))

    return TokenResponse(
        access_token=access,
        refresh_token=refresh,
        expires_in=ACCESS_TOKEN_EXPIRE_HOURS * 3600,
    )


@auth_router.post("/refresh", response_model=TokenResponse)
async def refresh_token(data: TokenRefresh, db: DBSession = Depends(get_db_dependency)):
    """Get a new access token using a refresh token."""
    payload = decode_token(data.refresh_token)

    if payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type (expected refresh token)",
        )

    user_id = payload.get("sub")
    store = UserStore(db)
    user = store.get_by_id(user_id)

    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    access = create_access_token(str(user.id), user.username)
    refresh = create_refresh_token(str(user.id))

    return TokenResponse(
        access_token=access,
        refresh_token=refresh,
        expires_in=ACCESS_TOKEN_EXPIRE_HOURS * 3600,
    )


@auth_router.get("/me", response_model=UserResponse)
async def get_me(user: User = Depends(get_current_user)):
    """Get current authenticated user info."""
    return UserResponse(
        user_id=str(user.id),
        username=user.username,
        email=user.email,
        full_name=user.full_name or "",
        created_at=user.created_at.isoformat(),
        is_active=user.is_active,
        onboarding_completed=getattr(user, "onboarding_completed", False) or False,
    )


@auth_router.post("/profile", response_model=UserResponse)
async def update_profile(
    data: OnboardingUpdate,
    user: User = Depends(get_current_user),
    db: DBSession = Depends(get_db_dependency),
):
    """Persist onboarding data and mark onboarding as completed."""
    user.onboarding_completed = data.onboarding_completed
    user.onboarding_data = data.onboarding_data
    db.add(user)
    db.commit()
    db.refresh(user)
    logger.info("Onboarding data saved for user %s", user.username)
    return UserResponse(
        user_id=str(user.id),
        username=user.username,
        email=user.email,
        full_name=user.full_name or "",
        created_at=user.created_at.isoformat(),
        is_active=user.is_active,
        onboarding_completed=user.onboarding_completed or False,
    )


@auth_router.post("/logout")
async def logout(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: DBSession = Depends(get_db_dependency),
    user: User = Depends(get_current_user),
):
    """Revoke the current access token."""
    payload = decode_token(credentials.credentials)
    jti = payload.get("jti")
    if jti:
        exp = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
        db.add(TokenBlacklist(jti=jti, user_id=user.id, token_type="access", expires_at=exp))
        db.commit()
    return {"status": "logged_out"}


@auth_router.post("/logout-all")
async def logout_all(
    db: DBSession = Depends(get_db_dependency),
    user: User = Depends(get_current_user),
):
    """Revoke all tokens for the current user (force re-login everywhere)."""
    # Blacklist all non-expired tokens is impractical without tracking all issued tokens.
    # Instead, bump a "tokens_invalidated_before" timestamp on the user.
    user.tokens_invalidated_at = datetime.now(timezone.utc)
    db.commit()
    return {"status": "all_sessions_revoked"}
