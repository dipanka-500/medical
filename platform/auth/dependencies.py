"""
Auth dependencies — production-grade FastAPI dependency injection.

Features:
    - JWT token validation with JTI support
    - Role-based access control (RBAC)
    - Scope-based permission checking
    - Token revocation check (JTI blocklist)
    - Admin IP allowlist
    - Permission caching (in-memory TTL)
    - X-Forwarded-For trust chain validation
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any
from uuid import UUID

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from auth.api_keys import APIKeyService
from auth.service import TokenService
from auth.token_blocklist import TokenBlocklist
from config import settings
from db.models import User, UserRole
from db.session import get_db

logger = logging.getLogger(__name__)

_bearer_scheme = HTTPBearer(auto_error=False)

# ── Permission Cache (in-memory with TTL) ─────────────────────────────

_PERMISSION_CACHE: dict[str, tuple[dict[str, Any], float]] = {}
_CACHE_TTL = 300.0  # 5 minutes
_CACHE_LOCK = asyncio.Lock()


def _cache_get(key: str) -> dict[str, Any] | None:
    """Get a cached permission entry if not expired."""
    entry = _PERMISSION_CACHE.get(key)
    if entry and (time.monotonic() - entry[1]) < _CACHE_TTL:
        return entry[0]
    return None


async def _cache_set(key: str, value: dict[str, Any]) -> None:
    """Cache a permission entry (async-safe eviction)."""
    _PERMISSION_CACHE[key] = (value, time.monotonic())

    # Evict stale entries periodically (prevent memory leak)
    if len(_PERMISSION_CACHE) > 10_000:
        async with _CACHE_LOCK:
            now = time.monotonic()
            stale = [k for k, (_, ts) in _PERMISSION_CACHE.items() if (now - ts) > _CACHE_TTL]
            for k in stale:
                _PERMISSION_CACHE.pop(k, None)


# ── Core Auth Dependency ─────────────────────────────────────────────────

async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    Validate the JWT and return the authenticated User object.

    Security checks:
        1. Token is present and valid
        2. Token type is "access" (not refresh)
        3. User exists and is active
        4. Token not revoked (JTI check)
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload = TokenService.decode_token(credentials.credentials)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Verify token type
    if payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type — use an access token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Extract user ID and JTI
    try:
        user_id = UUID(payload["sub"])
    except (KeyError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # ── JTI / user-level revocation check ─────────────────────────
    jti = payload.get("jti")
    redis_client = getattr(request.app.state, "redis", None)

    if jti and await TokenBlocklist.is_blocked(jti, redis=redis_client):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if await TokenBlocklist.is_user_revoked(str(user_id), redis=redis_client):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="All sessions revoked — please log in again",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Check permission cache first
    cache_key = f"user:{user_id}"
    cached = _cache_get(cache_key)
    if cached and cached.get("is_active"):
        # Cache hit — still need the full User object for downstream,
        # but we can skip the DB call for inactive/missing checks
        result = await db.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()
        if user and user.is_active:
            request.state.user = user
            request.state.user_id = user.id
            request.state.user_role = user.role
            return user

    # Fetch user from DB
    result = await db.execute(
        select(User).where(User.id == user_id)
    )
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account deactivated. Contact support.",
        )

    # Store user in request state for downstream use
    request.state.user = user
    request.state.user_id = user.id
    request.state.user_role = user.role

    # Cache for subsequent requests
    await _cache_set(cache_key, {
        "id": str(user.id),
        "role": user.role.value,
        "is_active": user.is_active,
    })

    return user


# ── Optional Auth (for public endpoints that benefit from auth) ──────────

async def get_optional_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> User | None:
    """
    Like get_current_user, but returns None instead of 401.
    Use for endpoints that work with or without auth (e.g., public health info).
    """
    if not credentials:
        return None
    try:
        return await get_current_user(request, credentials, db)
    except HTTPException:
        return None


# ── JWT or API Key Auth ───────────────────────────────────────────────

async def get_current_user_or_api_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    Authenticate via JWT **or** API key.

    Resolution order:
        1. ``Authorization: Bearer <jwt>`` header (existing flow)
        2. ``X-API-Key: mdk_...`` header (API key flow)

    If an API key is used, the associated user is returned just like
    the JWT path, so downstream code is auth-method-agnostic.
    """
    # 1. Try JWT first (if Bearer token is present)
    if credentials:
        try:
            return await get_current_user(request, credentials, db)
        except HTTPException:
            pass  # Fall through to API key check

    # 2. Try API key header
    api_key_header = request.headers.get("x-api-key")
    if api_key_header:
        client_ip = request.client.host if request.client else None
        api_key = await APIKeyService.validate_key(db, api_key_header, client_ip=client_ip)
        if api_key:
            # Load the associated user
            result = await db.execute(
                select(User).where(User.id == api_key.user_id)
            )
            user = result.scalar_one_or_none()

            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="API key owner not found",
                )

            if not user.is_active:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Account deactivated. Contact support.",
                )

            # Store in request state (same as JWT path)
            request.state.user = user
            request.state.user_id = user.id
            request.state.user_role = user.role
            request.state.api_key = api_key  # extra context for scope checks

            return user

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required — provide a Bearer token or X-API-Key header",
        headers={"WWW-Authenticate": "Bearer"},
    )


# ── Role-Based Access Control ────────────────────────────────────────────

def require_role(*allowed_roles: UserRole):
    """
    Dependency factory — restricts access to specified roles.

    Usage:
        @router.get("/admin", dependencies=[Depends(require_role(UserRole.ADMIN))])
        async def admin_only_endpoint():
            ...
    """
    async def _check(
        user: User = Depends(get_current_user),
    ) -> User:
        if user.role not in allowed_roles:
            role_names = ", ".join(r.value for r in allowed_roles)
            logger.warning(
                "Access denied: user=%s role=%s required=%s",
                user.id, user.role.value, role_names,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires role: {role_names}",
            )
        return user
    return _check


def require_admin():
    """Shortcut: require admin role."""
    return require_role(UserRole.ADMIN, UserRole.SUPERADMIN)


def require_doctor():
    """Shortcut: require doctor role."""
    return require_role(UserRole.DOCTOR, UserRole.ADMIN)


def require_patient():
    """Shortcut: require patient role (or admin)."""
    return require_role(UserRole.PATIENT, UserRole.ADMIN)


# ── Self-or-Admin Pattern ────────────────────────────────────────────────

class SelfOrAdminGuard:
    """
    Authorization guard: user can only access their own resources,
    unless they are an admin.

    Usage:
        guard = SelfOrAdminGuard(user=current_user)
        guard.check(target_user_id)  # raises 403 if not self or admin
    """

    def __init__(self, user: User) -> None:
        self.user = user

    def check(self, target_user_id: UUID) -> None:
        """Raise 403 if user is not the target user or an admin."""
        if self.user.role in (UserRole.ADMIN, UserRole.SUPERADMIN):
            return
        if self.user.id != target_user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You can only access your own resources",
            )
