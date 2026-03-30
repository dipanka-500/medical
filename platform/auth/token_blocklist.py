"""
Access-token JTI blocklist — Redis-backed with in-memory fallback.

When a user logs out or changes their password, we add the JTI of every
still-valid access token to this blocklist.  ``get_current_user`` checks
the blocklist on every request so that revoked access tokens are rejected
immediately (not just after they expire).

Redis keys use the pattern ``medai:jti:<jti>`` with TTL equal to the
remaining token lifetime, so entries self-clean.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from config import settings

logger = logging.getLogger(__name__)

_KEY_PREFIX = "medai:jti:"

# In-memory fallback when Redis is unavailable.  Bounded dict keyed by jti,
# valued by expiry (epoch seconds).  Periodically pruned on write.
_LOCAL_BLOCKLIST: dict[str, float] = {}
_LOCAL_MAX = 50_000


def _prune_local() -> None:
    """Remove expired entries from the in-memory fallback."""
    if len(_LOCAL_BLOCKLIST) < _LOCAL_MAX // 2:
        return
    now = time.time()
    expired = [k for k, exp in _LOCAL_BLOCKLIST.items() if exp <= now]
    for k in expired:
        _LOCAL_BLOCKLIST.pop(k, None)


class TokenBlocklist:
    """Check and add JTIs to the revocation blocklist."""

    @staticmethod
    async def add(jti: str, expires_in_seconds: int, redis: Any | None = None) -> None:
        """
        Block a JTI.  ``expires_in_seconds`` is the remaining lifetime of the
        access token so that the blocklist entry auto-expires with the token.
        """
        if expires_in_seconds <= 0:
            return  # Token already expired — nothing to block

        if redis:
            try:
                await redis.setex(f"{_KEY_PREFIX}{jti}", expires_in_seconds, "1")
                return
            except Exception as exc:
                logger.warning("Redis JTI blocklist write failed, using local fallback: %s", exc)

        # Fallback: in-memory
        _prune_local()
        _LOCAL_BLOCKLIST[jti] = time.time() + expires_in_seconds

    @staticmethod
    async def is_blocked(jti: str, redis: Any | None = None) -> bool:
        """Return True if the JTI has been revoked."""
        if redis:
            try:
                result = await redis.get(f"{_KEY_PREFIX}{jti}")
                if result is not None:
                    return True
                # Also check local in case Redis write failed earlier
            except Exception as exc:
                logger.debug("Redis JTI blocklist read failed: %s", exc)

        # Fallback: in-memory check
        exp = _LOCAL_BLOCKLIST.get(jti)
        if exp is not None:
            if exp > time.time():
                return True
            # Expired — clean it up
            _LOCAL_BLOCKLIST.pop(jti, None)

        return False

    @staticmethod
    async def revoke_user_tokens(
        user_id: str,
        redis: Any | None = None,
    ) -> None:
        """
        Best-effort revocation of all access tokens for a user.

        Since access tokens are stateless JWTs we cannot enumerate them.
        Instead, we store a "user-level revoke" marker.  ``is_blocked``
        checks both JTI-level and user-level markers.

        The marker TTL matches the max access-token lifetime so it
        self-cleans once all prior tokens have expired.
        """
        ttl = settings.jwt_access_token_expire_minutes * 60
        key = f"{_KEY_PREFIX}user:{user_id}"
        if redis:
            try:
                await redis.setex(key, ttl, "1")
                return
            except Exception as exc:
                logger.warning("Redis user-level revoke failed: %s", exc)
        _prune_local()
        _LOCAL_BLOCKLIST[f"user:{user_id}"] = time.time() + ttl

    @staticmethod
    async def is_user_revoked(user_id: str, redis: Any | None = None) -> bool:
        """Check whether ALL tokens for a user have been revoked."""
        key = f"{_KEY_PREFIX}user:{user_id}"
        if redis:
            try:
                result = await redis.get(key)
                if result is not None:
                    return True
            except Exception:
                pass

        exp = _LOCAL_BLOCKLIST.get(f"user:{user_id}")
        if exp is not None:
            if exp > time.time():
                return True
            _LOCAL_BLOCKLIST.pop(f"user:{user_id}", None)

        return False
