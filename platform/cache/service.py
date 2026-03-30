"""
Production-grade Redis caching service with graceful degradation.

Features:
    - Namespace-isolated keys (medai:cache: prefix)
    - JSON serialization for all values
    - Cache-aside pattern (get_or_set)
    - Pattern-based invalidation via SCAN (never KEYS)
    - Decorator for caching endpoint responses
    - Full graceful degradation: cache miss never crashes the app
"""

from __future__ import annotations

import functools
import json
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

KEY_PREFIX = "medai:cache:"


# ── Cache Key Builders ──────────────────────────────────────────────────


def session_list_key(user_id: str) -> str:
    """Cache key for a user's session list."""
    return f"sessions:user:{user_id}"


def session_messages_key(session_id: str) -> str:
    """Cache key for a session's messages."""
    return f"sessions:{session_id}:messages"


def user_profile_key(user_id: str) -> str:
    """Cache key for a user's profile."""
    return f"user:profile:{user_id}"


def health_status_key() -> str:
    """Cache key for the deep health check result."""
    return "health:deep"


# ── CacheService ────────────────────────────────────────────────────────


class CacheService:
    """
    Async Redis caching service with graceful degradation.

    If the Redis client is ``None`` or unavailable, every operation
    degrades silently — reads return ``None``, writes are no-ops.
    Cache failures never propagate to callers.
    """

    def __init__(self, redis_client: Any | None = None) -> None:
        self._redis = redis_client

    # ── helpers ──────────────────────────────────────────────────

    def _prefixed(self, key: str) -> str:
        return f"{KEY_PREFIX}{key}"

    @property
    def available(self) -> bool:
        return self._redis is not None

    # ── core operations ─────────────────────────────────────────

    async def get(self, key: str) -> Any | None:
        """Get a value from cache and deserialize from JSON."""
        if not self.available:
            return None
        try:
            raw = await self._redis.get(self._prefixed(key))
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as exc:
            logger.debug("Cache GET failed for %s: %s", key, exc)
            return None

    async def set(self, key: str, value: Any, ttl: int = 300) -> None:
        """Serialize value to JSON and store with TTL (seconds)."""
        if not self.available:
            return
        try:
            serialized = json.dumps(value, default=str)
            await self._redis.setex(self._prefixed(key), ttl, serialized)
        except Exception as exc:
            logger.debug("Cache SET failed for %s: %s", key, exc)

    async def delete(self, key: str) -> None:
        """Delete a single key from cache."""
        if not self.available:
            return
        try:
            await self._redis.delete(self._prefixed(key))
        except Exception as exc:
            logger.debug("Cache DELETE failed for %s: %s", key, exc)

    async def delete_pattern(self, pattern: str) -> None:
        """
        Delete all keys matching a pattern.

        Uses SCAN to iterate keys (never KEYS) to avoid blocking Redis
        on large keyspaces.
        """
        if not self.available:
            return
        try:
            full_pattern = self._prefixed(pattern)
            cursor = 0
            while True:
                cursor, keys = await self._redis.scan(
                    cursor=cursor, match=full_pattern, count=100,
                )
                if keys:
                    await self._redis.delete(*keys)
                if cursor == 0:
                    break
        except Exception as exc:
            logger.debug("Cache DELETE_PATTERN failed for %s: %s", pattern, exc)

    async def get_or_set(
        self,
        key: str,
        factory: Callable[..., Any],
        ttl: int = 300,
    ) -> Any:
        """
        Cache-aside pattern.

        Returns the cached value if present, otherwise calls *factory*
        (which must be an async callable), stores the result, and returns it.
        If the cache is unavailable the factory is always called.
        """
        cached = await self.get(key)
        if cached is not None:
            return cached

        value = await factory()

        await self.set(key, value, ttl=ttl)
        return value

    async def invalidate_user_cache(self, user_id: str) -> None:
        """Clear all cached data for a specific user."""
        await self.delete(session_list_key(user_id))
        await self.delete(user_profile_key(user_id))
        # Also sweep any pattern-based keys for the user
        await self.delete_pattern(f"*:user:{user_id}*")

    # ── decorator ───────────────────────────────────────────────

    def cache_response(self, key: str, ttl: int = 300):
        """
        Decorator for caching the return value of an async endpoint.

        Usage::

            @cache_service.cache_response("my:key", ttl=60)
            async def my_endpoint():
                ...
        """
        def decorator(func: Callable):
            @functools.wraps(func)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                cached = await self.get(key)
                if cached is not None:
                    return cached
                result = await func(*args, **kwargs)
                await self.set(key, result, ttl=ttl)
                return result
            return wrapper
        return decorator
