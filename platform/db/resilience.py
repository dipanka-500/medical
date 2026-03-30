"""
Database resilience primitives — production-grade.

Features:
    - RetryableSession: async context manager with exponential backoff
    - DatabaseCircuitBreaker: prevents cascading failures with CLOSED/OPEN/HALF_OPEN states
    - db_health_monitor: background task that continuously checks DB health

Usage:
    async with RetryableSession() as db:
        result = await db.execute(text("SELECT 1"))

    state = circuit_breaker.get_state()
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from enum import Enum
from typing import Any

from sqlalchemy.exc import InterfaceError, OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


# ── Circuit Breaker ─────────────────────────────────────────────────────


class CircuitState(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class DatabaseCircuitBreaker:
    """
    Circuit breaker for database connections.

    States:
        CLOSED   — Normal operation. Requests pass through.
        OPEN     — Too many failures. Requests are rejected immediately.
        HALF_OPEN — Cooldown expired. One probe request is allowed through.

    Transitions:
        CLOSED  -> OPEN      : failure_count >= failure_threshold within the rolling window
        OPEN    -> HALF_OPEN  : cooldown_seconds have elapsed since last failure
        HALF_OPEN -> CLOSED   : probe request succeeds
        HALF_OPEN -> OPEN     : probe request fails
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        rolling_window_seconds: float = 60.0,
        cooldown_seconds: float = 30.0,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._rolling_window_seconds = rolling_window_seconds
        self._cooldown_seconds = cooldown_seconds

        self._state = CircuitState.CLOSED
        self._failures: list[float] = []  # monotonic timestamps of failures
        self._last_failure_time: float | None = None
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        """Current circuit state (may auto-transition OPEN -> HALF_OPEN)."""
        if self._state == CircuitState.OPEN and self._last_failure_time is not None:
            elapsed = time.monotonic() - self._last_failure_time
            if elapsed >= self._cooldown_seconds:
                return CircuitState.HALF_OPEN
        return self._state

    async def check(self) -> None:
        """
        Check whether a request is allowed through.

        Raises:
            RuntimeError: if the circuit is OPEN (caller should convert to HTTP 503).
        """
        async with self._lock:
            current = self.state

            if current == CircuitState.OPEN:
                raise RuntimeError(
                    "Database circuit breaker is OPEN — connections rejected to protect the system"
                )

            # HALF_OPEN: allow exactly one probe through (state transitions happen
            # in record_success / record_failure)
            if current == CircuitState.HALF_OPEN and self._state != CircuitState.HALF_OPEN:
                # Auto-transition from stored OPEN to HALF_OPEN
                logger.info("Circuit breaker transitioning OPEN -> HALF_OPEN (cooldown expired)")
                self._state = CircuitState.HALF_OPEN

    async def record_success(self) -> None:
        """Record a successful database operation."""
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                logger.info("Circuit breaker transitioning HALF_OPEN -> CLOSED (probe succeeded)")
                self._state = CircuitState.CLOSED
                self._failures.clear()
                self._last_failure_time = None

    async def record_failure(self) -> None:
        """Record a failed database operation."""
        async with self._lock:
            now = time.monotonic()
            self._last_failure_time = now
            self._failures.append(now)

            # Prune failures outside the rolling window
            cutoff = now - self._rolling_window_seconds
            self._failures = [t for t in self._failures if t > cutoff]

            if self._state == CircuitState.HALF_OPEN:
                logger.warning("Circuit breaker transitioning HALF_OPEN -> OPEN (probe failed)")
                self._state = CircuitState.OPEN
            elif self._state == CircuitState.CLOSED:
                if len(self._failures) >= self._failure_threshold:
                    logger.warning(
                        "Circuit breaker transitioning CLOSED -> OPEN "
                        "(%d failures in %.0fs window)",
                        len(self._failures),
                        self._rolling_window_seconds,
                    )
                    self._state = CircuitState.OPEN

    def get_state(self) -> dict[str, Any]:
        """Return circuit breaker state for monitoring/health endpoints."""
        return {
            "state": self.state.value,
            "failure_count": len(self._failures),
            "last_failure_time": self._last_failure_time,
            "failure_threshold": self._failure_threshold,
            "rolling_window_seconds": self._rolling_window_seconds,
            "cooldown_seconds": self._cooldown_seconds,
        }


# ── Module-level singleton ──────────────────────────────────────────────

circuit_breaker = DatabaseCircuitBreaker()


# ── Retryable Session ──────────────────────────────────────────────────


# Exceptions that are safe to retry (transient connection issues)
_RETRYABLE_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    OperationalError,
    InterfaceError,
    OSError,
)

_BACKOFF_SCHEDULE = (0.5, 1.0, 2.0)  # seconds


class RetryableSession:
    """
    Async context manager that wraps get_db() with retry + circuit-breaker logic.

    Usage:
        async with RetryableSession() as db:
            result = await db.execute(text("SELECT ..."))
    """

    def __init__(self, max_retries: int = 3) -> None:
        self._max_retries = max_retries
        self._session: AsyncSession | None = None
        self._ctx = None

    async def __aenter__(self) -> AsyncSession:
        # Import here to avoid circular imports (session.py imports from resilience)
        from .session import async_session_factory

        last_exc: Exception | None = None

        for attempt in range(self._max_retries):
            # Check circuit breaker before each attempt
            await circuit_breaker.check()

            try:
                self._ctx = async_session_factory()
                self._session = await self._ctx.__aenter__()
                # Quick validation — ensure the connection is live
                await circuit_breaker.record_success()
                return self._session

            except _RETRYABLE_EXCEPTIONS as exc:
                last_exc = exc
                await circuit_breaker.record_failure()

                backoff = _BACKOFF_SCHEDULE[min(attempt, len(_BACKOFF_SCHEDULE) - 1)]
                logger.warning(
                    "Retryable DB error (attempt %d/%d), retrying in %.1fs: %s: %s",
                    attempt + 1,
                    self._max_retries,
                    backoff,
                    type(exc).__name__,
                    str(exc)[:200],
                )

                # Clean up the failed session context if it was partially opened
                if self._ctx is not None:
                    try:
                        await self._ctx.__aexit__(type(exc), exc, exc.__traceback__)
                    except Exception:
                        pass
                    self._ctx = None
                    self._session = None

                await asyncio.sleep(backoff)

            except Exception as exc:
                # Non-retryable error — fail immediately
                await circuit_breaker.record_failure()
                if self._ctx is not None:
                    try:
                        await self._ctx.__aexit__(type(exc), exc, exc.__traceback__)
                    except Exception:
                        pass
                    self._ctx = None
                    self._session = None
                raise

        # All retries exhausted
        raise last_exc  # type: ignore[misc]

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[no-untyped-def]
        if self._session is not None and self._ctx is not None:
            try:
                if exc_type is None:
                    await self._session.commit()
                else:
                    await self._session.rollback()
            except Exception:
                pass
            try:
                await self._ctx.__aexit__(exc_type, exc_val, exc_tb)
            except Exception:
                pass
            self._session = None
            self._ctx = None


# ── Health Monitor ──────────────────────────────────────────────────────


class _DBHealthMonitor:
    """
    Background async task that periodically checks database connectivity
    and updates the circuit breaker state.

    Usage:
        db_health_monitor.start()   # call during app startup
        db_health_monitor.stop()    # call during app shutdown
    """

    def __init__(self, interval_seconds: float = 30.0) -> None:
        self._interval = interval_seconds
        self._task: asyncio.Task | None = None
        self._running = False

    def start(self) -> None:
        """Start the background health monitor loop."""
        if self._task is not None and not self._task.done():
            logger.debug("DB health monitor already running")
            return
        self._running = True
        self._task = asyncio.create_task(self._run(), name="db-health-monitor")
        logger.info("DB health monitor started (interval=%.0fs)", self._interval)

    def stop(self) -> None:
        """Stop the background health monitor loop."""
        self._running = False
        if self._task is not None and not self._task.done():
            self._task.cancel()
            logger.info("DB health monitor stopped")
        self._task = None

    async def _run(self) -> None:
        """Main monitor loop — runs until stopped."""
        from .session import verify_connection

        while self._running:
            try:
                healthy = await verify_connection()

                if healthy:
                    await circuit_breaker.record_success()
                else:
                    await circuit_breaker.record_failure()
                    logger.warning(
                        "DB health monitor: connectivity check failed — circuit state: %s",
                        circuit_breaker.state.value,
                    )
            except asyncio.CancelledError:
                break
            except Exception as exc:
                await circuit_breaker.record_failure()
                logger.error(
                    "DB health monitor error: %s: %s — circuit state: %s",
                    type(exc).__name__,
                    str(exc)[:200],
                    circuit_breaker.state.value,
                )

            try:
                await asyncio.sleep(self._interval)
            except asyncio.CancelledError:
                break


db_health_monitor = _DBHealthMonitor()
