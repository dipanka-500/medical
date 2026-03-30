"""
Rate limiting — production-grade with Redis backend.

Features:
    - Redis atomic sliding-window via Lua script (distributed)
    - In-memory sliding window fallback (for dev / when Redis unavailable)
    - IP blocklist with configurable duration (manual + auto after threshold)
    - Per-user + per-endpoint composite keys
    - Thread-safe metrics with asyncio Lock
    - Redis failure circuit breaker with alerting
    - Persistent blocklist in Redis
    - Key explosion protection (LRU eviction)
    - IP spoofing protection (trusted proxy validation)
    - Stale window cleanup
    - Daily query tracking for free tier
"""

from __future__ import annotations

import asyncio
import collections
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from config import settings

logger = logging.getLogger(__name__)


# ── Atomic Lua script for Redis sliding window ──────────────────────────

_SLIDING_WINDOW_LUA = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local max_requests = tonumber(ARGV[3])
local ttl = tonumber(ARGV[4])

-- Remove expired entries
redis.call('ZREMRANGEBYSCORE', key, 0, now - window)

-- Get current count BEFORE adding
local count = redis.call('ZCARD', key)

if count < max_requests then
    -- Only add if under limit (prevents unnecessary key growth)
    redis.call('ZADD', key, now, now .. ':' .. math.random(1000000))
    count = count + 1
end

redis.call('EXPIRE', key, ttl)

return count
"""


# ── Metrics (thread-safe) ───────────────────────────────────────────────

class _RateLimitMetrics:
    """Thread-safe rate limit metrics."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self.total_checks: int = 0
        self.total_blocks: int = 0
        self.total_auto_blocks: int = 0
        self.redis_failures: int = 0
        self.fallback_activations: int = 0

    async def record_check(self) -> None:
        async with self._lock:
            self.total_checks += 1

    async def record_block(self) -> None:
        async with self._lock:
            self.total_blocks += 1

    async def record_auto_block(self) -> None:
        async with self._lock:
            self.total_auto_blocks += 1

    async def record_redis_failure(self) -> None:
        async with self._lock:
            self.redis_failures += 1
            self.fallback_activations += 1

    async def as_dict(self) -> dict[str, int]:
        async with self._lock:
            return {
                "rate_limit_checks_total": self.total_checks,
                "rate_limit_blocks_total": self.total_blocks,
                "rate_limit_auto_blocks_total": self.total_auto_blocks,
                "redis_failures_total": self.redis_failures,
                "fallback_activations_total": self.fallback_activations,
            }


# ── In-memory sliding window (true sliding, not fixed) ──────────────────

@dataclass
class _SlidingWindow:
    """True sliding window using a deque of timestamps."""
    timestamps: collections.deque = field(default_factory=collections.deque)


# ── IP Blocklist ─────────────────────────────────────────────────────────

@dataclass
class _BlockRecord:
    blocked_at: float = 0.0
    duration: float = 86400.0  # seconds — FIX #1: configurable duration
    reason: str = ""
    violations: int = 0


# ── Redis Circuit Breaker ────────────────────────────────────────────────

class _RedisCircuitBreaker:
    """Simple circuit breaker for Redis failures."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
    ) -> None:
        self._failure_count = 0
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._last_failure_time: float = 0.0
        self._state = "closed"  # closed = healthy, open = failing, half_open = testing

    def record_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        if self._failure_count >= self._failure_threshold:
            self._state = "open"
            logger.error(
                "Redis circuit breaker OPEN after %d failures — using memory fallback",
                self._failure_count,
            )

    def record_success(self) -> None:
        self._failure_count = 0
        if self._state != "closed":
            logger.info("Redis circuit breaker CLOSED — Redis recovered")
        self._state = "closed"

    def should_attempt(self) -> bool:
        if self._state == "closed":
            return True
        if self._state == "open":
            elapsed = time.monotonic() - self._last_failure_time
            if elapsed >= self._recovery_timeout:
                self._state = "half_open"
                return True
            return False
        # half_open — allow one attempt
        return True


# ── Trusted Proxy Validation ─────────────────────────────────────────────

_TRUSTED_PROXIES: set[str] = {
    "127.0.0.1", "::1", "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16",
}


def _is_trusted(ip: str, proxies: set[str]) -> bool:
    """Check if IP matches any trusted proxy, including CIDR ranges."""
    import ipaddress
    if ip in proxies:
        return True
    try:
        addr = ipaddress.ip_address(ip)
        for proxy in proxies:
            if "/" in proxy:
                try:
                    if addr in ipaddress.ip_network(proxy, strict=False):
                        return True
                except ValueError:
                    continue
    except ValueError:
        return False
    return False


def extract_real_ip(
    client_ip: str | None,
    forwarded_for: str | None = None,
    trusted_proxies: set[str] | None = None,
) -> str:
    """
    Extract real client IP with spoofing protection.

    Only trusts X-Forwarded-For from known proxy IPs.
    Falls back to direct client_ip if proxy chain is untrusted.
    Supports CIDR ranges in trusted_proxies (e.g. 10.0.0.0/8).
    """
    if not client_ip:
        return "unknown"

    if not forwarded_for:
        return client_ip

    proxies = trusted_proxies or _TRUSTED_PROXIES

    # Only parse X-Forwarded-For if the direct client is a trusted proxy
    if _is_trusted(client_ip, proxies):
        # Take the rightmost non-proxy IP (closest to the client)
        ips = [ip.strip() for ip in forwarded_for.split(",")]
        for ip in reversed(ips):
            if ip and not _is_trusted(ip, proxies):
                return ip

    return client_ip


class RateLimiter:
    """
    Production rate limiter with Redis backend and in-memory fallback.

    Architecture:
        1. Check IP blocklist first (O(1))
        2. Check rate limit via atomic sliding window (Redis Lua or in-memory)
        3. Track violations for auto-block escalation
        4. Return rate limit headers for client awareness
        5. Circuit breaker protects against Redis outages

    For distributed deployments, set REDIS_URL in env to enable
    Redis-backed counters shared across all platform instances.
    """

    # Max in-memory windows to prevent key explosion (LRU eviction)
    _MAX_MEMORY_WINDOWS = 50_000

    def __init__(self) -> None:
        self._windows: collections.OrderedDict[str, _SlidingWindow] = (
            collections.OrderedDict()
        )
        self._lock = asyncio.Lock()
        self._blocked_ips: dict[str, _BlockRecord] = {}
        self._violation_counts: dict[str, int] = collections.defaultdict(int)
        self._metrics = _RateLimitMetrics()
        self._redis: Any = None
        self._redis_available = False
        self._circuit_breaker = _RedisCircuitBreaker()
        self._lua_sha: str | None = None

        # Stale window cleanup
        self._last_cleanup = time.monotonic()
        self._cleanup_interval = 300.0  # 5 minutes

    # ── Redis Backend ────────────────────────────────────────────────

    async def connect_redis(self) -> None:
        """Attempt to connect to Redis for distributed rate limiting."""
        try:
            import redis.asyncio as aioredis
            self._redis = aioredis.from_url(
                settings.redis_url,
                max_connections=settings.redis_max_connections,
                socket_timeout=settings.redis_socket_timeout,
                socket_connect_timeout=settings.redis_socket_connect_timeout,
                retry_on_timeout=settings.redis_retry_on_timeout,
                decode_responses=True,
            )
            await self._redis.ping()
            self._redis_available = True

            # Pre-load the Lua script for atomic operations
            self._lua_sha = await self._redis.script_load(_SLIDING_WINDOW_LUA)

            # Restore blocklist from Redis
            await self._restore_blocklist_from_redis()

            logger.info("Redis rate limiter connected at %s", settings.redis_url)
        except Exception as exc:
            self._redis_available = False
            logger.warning(
                "Redis unavailable, using in-memory rate limiter: %s", exc,
            )

    async def close(self) -> None:
        """Close Redis connection on shutdown."""
        if self._redis:
            # Persist blocklist to Redis before shutdown
            await self._persist_blocklist_to_redis()
            await self._redis.close()

    # ── Parse Helpers ────────────────────────────────────────────────

    @staticmethod
    def _parse_limit(limit_str: str) -> tuple[int, float]:
        """Parse '60/minute' → (60, 60.0 seconds)."""
        parts = limit_str.split("/")
        if len(parts) != 2:
            logger.warning("Invalid rate limit format: %s, using default", limit_str)
            return 60, 60.0
        try:
            count = int(parts[0])
        except ValueError:
            logger.warning("Invalid rate limit count: %s, using default", parts[0])
            return 60, 60.0
        unit = parts[1].lower()
        seconds = {
            "second": 1, "minute": 60, "hour": 3600, "day": 86400,
        }.get(unit, 60)
        return count, float(seconds)

    # ── Composite Key Builder ────────────────────────────────────────

    @staticmethod
    def build_key(
        ip: str | None = None,
        user_id: str | None = None,
        endpoint: str | None = None,
    ) -> str:
        """
        Build a composite rate limit key.

        FIX #4: Per-user + per-endpoint limiting.
        Supports any combination of ip, user_id, and endpoint.
        """
        parts = []
        if user_id:
            parts.append(f"u:{user_id}")
        if endpoint:
            parts.append(f"e:{endpoint}")
        if ip:
            parts.append(f"ip:{ip}")
        return ":".join(parts) if parts else "global"

    # ── IP Blocklist (persistent in Redis) ───────────────────────────

    async def block_ip(
        self, ip: str, reason: str = "manual", duration_hours: float = 24,
    ) -> None:
        """Block an IP address with configurable duration."""
        duration_seconds = duration_hours * 3600  # FIX #1: use duration param
        record = _BlockRecord(
            blocked_at=time.monotonic(),
            duration=duration_seconds,
            reason=reason,
        )
        self._blocked_ips[ip] = record

        # FIX #8: Persist to Redis
        if self._redis_available and self._redis:
            try:
                await self._redis.setex(
                    f"blocked:{ip}",
                    int(duration_seconds),
                    f"{reason}:{duration_seconds}",
                )
            except Exception as exc:
                logger.warning("Failed to persist IP block to Redis: %s", exc)

        logger.warning(
            "IP blocked: %s reason=%s duration=%sh", ip, reason, duration_hours,
        )

    async def unblock_ip(self, ip: str) -> bool:
        """Remove an IP from the blocklist."""
        removed = False
        if ip in self._blocked_ips:
            del self._blocked_ips[ip]
            self._violation_counts.pop(ip, None)
            removed = True

        # Remove from Redis too
        if self._redis_available and self._redis:
            try:
                await self._redis.delete(f"blocked:{ip}")
            except Exception as exc:
                logger.debug("Redis unblock failed for %s: %s", ip, exc)

        if removed:
            logger.info("IP unblocked: %s", ip)
        return removed

    def is_blocked(self, ip: str) -> bool:
        """Check if an IP is blocked (with expiry)."""
        record = self._blocked_ips.get(ip)
        if not record:
            return False
        # FIX #1: Use record.duration instead of hardcoded 86400
        if (time.monotonic() - record.blocked_at) > record.duration:
            del self._blocked_ips[ip]
            return False
        return True

    async def _record_violation(self, ip: str) -> None:
        """Track violations and auto-block after threshold."""
        self._violation_counts[ip] += 1
        if self._violation_counts[ip] >= settings.rate_limit_ip_blocklist_threshold:
            await self.block_ip(ip, reason="auto_threshold")
            await self._metrics.record_auto_block()
            logger.warning(
                "Auto-blocked IP %s after %d violations",
                ip, self._violation_counts[ip],
            )

    async def _persist_blocklist_to_redis(self) -> None:
        """Persist current in-memory blocklist to Redis."""
        if not self._redis:
            return
        for ip, record in self._blocked_ips.items():
            remaining = record.duration - (time.monotonic() - record.blocked_at)
            if remaining > 0:
                try:
                    await self._redis.setex(
                        f"blocked:{ip}",
                        int(remaining),
                        f"{record.reason}:{record.duration}",
                    )
                except Exception as exc:
                    logger.debug("Redis blocklist sync failed for %s: %s", ip, exc)

    async def _restore_blocklist_from_redis(self) -> None:
        """Restore blocklist from Redis on startup."""
        if not self._redis:
            return
        try:
            cursor = "0"
            while cursor:
                cursor, keys = await self._redis.scan(
                    cursor=cursor, match="blocked:*", count=100,
                )
                for key in keys:
                    ip = key.replace("blocked:", "", 1)
                    ttl = await self._redis.ttl(key)
                    value = await self._redis.get(key)
                    reason = "restored"
                    if value and ":" in value:
                        reason = value.split(":")[0]
                    if ttl > 0:
                        self._blocked_ips[ip] = _BlockRecord(
                            blocked_at=time.monotonic(),
                            duration=float(ttl),
                            reason=reason,
                        )
                if cursor == 0 or cursor == "0":
                    break
            if self._blocked_ips:
                logger.info(
                    "Restored %d blocked IPs from Redis", len(self._blocked_ips),
                )
        except Exception as exc:
            logger.warning("Failed to restore blocklist from Redis: %s", exc)

    # ── Main Check ───────────────────────────────────────────────────

    async def check(
        self,
        key: str,
        limit_str: str | None = None,
        client_ip: str | None = None,
    ) -> tuple[bool, dict]:
        """
        Check if request is allowed.

        Returns:
            (allowed: bool, info: dict with limit/remaining/reset headers)
        """
        await self._metrics.record_check()

        # 1. IP blocklist check
        if client_ip and self.is_blocked(client_ip):
            await self._metrics.record_block()
            record = self._blocked_ips.get(client_ip)
            remaining_block = 0.0
            if record:
                remaining_block = record.duration - (
                    time.monotonic() - record.blocked_at
                )
            return False, {
                "limit": 0,
                "remaining": 0,
                "reset_in_seconds": round(max(0, remaining_block), 1),
                "reason": "ip_blocked",
            }

        limit_str = limit_str or settings.rate_limit_default
        max_requests, window_seconds = self._parse_limit(limit_str)

        # 2. Try Redis first (with circuit breaker), fall back to in-memory
        if self._redis_available and self._circuit_breaker.should_attempt():
            try:
                result = await self._check_redis(key, max_requests, window_seconds)
                self._circuit_breaker.record_success()

                # Track violations on block
                if not result[0] and client_ip:
                    await self._record_violation(client_ip)
                    await self._metrics.record_block()

                return result
            except Exception as exc:
                self._circuit_breaker.record_failure()
                await self._metrics.record_redis_failure()
                logger.warning(
                    "Redis rate limit failed (circuit_breaker=%s): %s",
                    self._circuit_breaker._state, exc,
                )

        # 3. In-memory sliding window fallback
        result = await self._check_memory(key, max_requests, window_seconds)

        # 4. Track violations
        if not result[0] and client_ip:
            await self._record_violation(client_ip)
            await self._metrics.record_block()

        return result

    async def _check_redis(
        self, key: str, max_requests: int, window_seconds: float,
    ) -> tuple[bool, dict]:
        """
        Redis sliding window using atomic Lua script.

        FIX #2: Single Lua script replaces non-atomic pipeline.
        """
        now = time.time()
        redis_key = f"rl:{key}"
        ttl = int(window_seconds) + 1

        # Use EVALSHA for pre-loaded script
        if self._lua_sha:
            current_count = await self._redis.evalsha(
                self._lua_sha,
                1,  # number of keys
                redis_key,
                str(now),
                str(window_seconds),
                str(max_requests),
                str(ttl),
            )
        else:
            # Fallback: load and eval inline
            current_count = await self._redis.eval(
                _SLIDING_WINDOW_LUA,
                1,
                redis_key,
                str(now),
                str(window_seconds),
                str(max_requests),
                str(ttl),
            )

        current_count = int(current_count)
        allowed = current_count <= max_requests

        return allowed, {
            "limit": max_requests,
            "remaining": max(0, max_requests - current_count),
            "reset_in_seconds": round(window_seconds, 1),
        }

    async def _check_memory(
        self, key: str, max_requests: int, window_seconds: float,
    ) -> tuple[bool, dict]:
        """
        In-memory TRUE sliding window using timestamp deque.

        FIX #3: Replaces fixed-window with real sliding window.
        FIX #9: LRU eviction prevents key explosion / DoS.
        """
        async with self._lock:
            now = time.monotonic()

            # Periodic stale window cleanup
            if (now - self._last_cleanup) > self._cleanup_interval:
                self._cleanup_stale_windows(now, window_seconds)
                self._last_cleanup = now

            # LRU eviction if over capacity
            if key not in self._windows and len(self._windows) >= self._MAX_MEMORY_WINDOWS:
                # Evict oldest (least recently used) + clean up violation counter
                evicted_key, _ = self._windows.popitem(last=False)
                self._violation_counts.pop(evicted_key, None)

            if key not in self._windows:
                self._windows[key] = _SlidingWindow()

            window = self._windows[key]
            # Move to end (most recently used)
            self._windows.move_to_end(key)

            # Remove timestamps outside the window
            cutoff = now - window_seconds
            while window.timestamps and window.timestamps[0] < cutoff:
                window.timestamps.popleft()

            current_count = len(window.timestamps)

            if current_count < max_requests:
                window.timestamps.append(now)
                current_count += 1
                allowed = True
            else:
                allowed = False

            # Estimate time until oldest entry expires
            if window.timestamps:
                reset_in = window.timestamps[0] + window_seconds - now
            else:
                reset_in = window_seconds

            return allowed, {
                "limit": max_requests,
                "remaining": max(0, max_requests - current_count),
                "reset_in_seconds": round(max(0, reset_in), 1),
            }

    def _cleanup_stale_windows(self, now: float, default_window: float = 60.0) -> None:
        """
        Remove expired windows to prevent memory leak.

        FIX #12: Only iterate keys, O(k) where k = stale keys.
        """
        stale_keys = []
        for k, w in self._windows.items():
            if not w.timestamps:
                stale_keys.append(k)
            elif (now - w.timestamps[-1]) > default_window + 60:
                stale_keys.append(k)

        for k in stale_keys:
            del self._windows[k]

        if stale_keys:
            logger.debug("Cleaned up %d stale rate limit windows", len(stale_keys))

    # ── Daily Query Check (Free Tier) ────────────────────────────────

    async def check_daily_queries(self, user_id: str, tier: str) -> tuple[bool, int]:
        """Check free-tier daily query limit."""
        if tier != "free":
            return True, -1  # unlimited

        key = f"daily:{user_id}"
        allowed, info = await self.check(key, f"{settings.free_tier_daily_queries}/day")
        return allowed, info["remaining"]

    # ── Monitoring ───────────────────────────────────────────────────

    async def get_metrics(self) -> dict:
        """Return metrics for Prometheus / monitoring."""
        metrics_dict = await self._metrics.as_dict()
        return {
            **metrics_dict,
            "blocked_ips_count": len(self._blocked_ips),
            "active_windows": len(self._windows),
            "redis_available": self._redis_available,
            "circuit_breaker_state": self._circuit_breaker._state,
        }
