"""
Concurrency control, backpressure, and load shedding for the MedAI Platform.

Production-grade mechanisms:
    - AIQuerySemaphore: bounds concurrent AI requests with queue-depth load shedding
    - BackpressureMiddleware: caps total active HTTP requests platform-wide
    - GracefulDegradation: monitors system health and auto-degrades/recovers
"""

from __future__ import annotations

import asyncio
import enum
import logging
import time
from typing import Any

import psutil
from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

logger = logging.getLogger(__name__)

# Health-check paths exempt from backpressure limits
_HEALTH_PATHS = frozenset({
    "/api/v1/health",
    "/api/v1/health/ready",
    "/api/v1/health/live",
    "/healthz",
    "/readyz",
    "/livez",
})


# ── AIQuerySemaphore ─────────────────────────────────────────────────────


class AIQuerySemaphore:
    """
    Controls concurrent AI query execution with queue-depth load shedding.

    When the waiting queue exceeds 2x max_concurrent, new requests are
    rejected immediately with HTTP 503 (load shedding) rather than piling
    up and increasing tail latency for everyone.
    """

    def __init__(
        self,
        max_concurrent: int = 10,
        queue_timeout: float = 30.0,
    ) -> None:
        self._max_concurrent = max_concurrent
        self._queue_timeout = queue_timeout
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._active: int = 0
        self._waiting: int = 0
        self._lock = asyncio.Lock()

    # ── public API ───────────────────────────────────────────────

    async def acquire(self) -> None:
        """
        Acquire a slot for an AI query.

        Raises:
            OverflowError: if queue depth exceeds 2x max_concurrent (load shedding).
            TimeoutError: if the slot is not available within queue_timeout seconds.
        """
        async with self._lock:
            # Load shedding: reject if queue is already too deep
            if self._waiting >= self._max_concurrent * 2:
                raise OverflowError(
                    f"AI query queue is full ({self._waiting} waiting, "
                    f"max {self._max_concurrent * 2}). Try again later."
                )
            self._waiting += 1

        try:
            await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=self._queue_timeout,
            )
        except asyncio.TimeoutError:
            async with self._lock:
                self._waiting -= 1
            raise TimeoutError(
                f"Timed out waiting for AI query slot after {self._queue_timeout}s. "
                "The system is under heavy load."
            )

        async with self._lock:
            self._waiting -= 1
            self._active += 1

    async def release(self) -> None:
        """Release an AI query slot."""
        async with self._lock:
            self._active = max(0, self._active - 1)
        self._semaphore.release()

    def get_stats(self) -> dict[str, Any]:
        """Return current concurrency statistics."""
        utilization = (self._active / self._max_concurrent * 100) if self._max_concurrent else 0
        return {
            "active": self._active,
            "waiting": self._waiting,
            "max_concurrent": self._max_concurrent,
            "utilization_pct": round(utilization, 1),
        }


# ── BackpressureMiddleware ───────────────────────────────────────────────


class BackpressureMiddleware(BaseHTTPMiddleware):
    """
    Limits total active requests across all endpoints.

    When the platform is at capacity, returns 503 with a Retry-After
    header so clients can implement exponential back-off. Health-check
    endpoints are always exempt.
    """

    def __init__(self, app: Any, max_active_requests: int = 200) -> None:
        super().__init__(app)
        self._max_active = max_active_requests
        self._active: int = 0
        self._lock = asyncio.Lock()

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint,
    ) -> Response:
        # Health checks are always allowed
        if request.url.path in _HEALTH_PATHS:
            return await call_next(request)

        async with self._lock:
            if self._active >= self._max_active:
                logger.warning(
                    "Backpressure: rejecting request to %s (%d/%d active)",
                    request.url.path, self._active, self._max_active,
                )
                return JSONResponse(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    content={
                        "detail": "Server is at capacity. Please retry shortly.",
                    },
                    headers={
                        "Retry-After": "5",
                        "X-Active-Requests": str(self._active),
                    },
                )
            self._active += 1

        try:
            response = await call_next(request)
            response.headers["X-Active-Requests"] = str(self._active)
            return response
        finally:
            async with self._lock:
                self._active = max(0, self._active - 1)


# ── GracefulDegradation ──────────────────────────────────────────────────


class DegradationMode(str, enum.Enum):
    NORMAL = "normal"
    DEGRADED = "degraded"
    CRITICAL = "critical"


class GracefulDegradation:
    """
    Monitors system health signals and automatically adjusts operating mode.

    Modes:
        NORMAL   — all features enabled.
        DEGRADED — non-essential features disabled (search/RAG, streaming).
        CRITICAL — reject new AI queries, return cached responses only.

    The class tracks three signals:
        - Memory usage (% of system RAM)
        - Request latency (rolling average, seconds)
        - Error rate (rolling ratio 0.0–1.0)

    Thresholds can be tuned via constructor parameters.
    """

    def __init__(
        self,
        *,
        memory_degraded_pct: float = 80.0,
        memory_critical_pct: float = 95.0,
        latency_degraded_s: float = 5.0,
        latency_critical_s: float = 15.0,
        error_rate_degraded: float = 0.10,
        error_rate_critical: float = 0.30,
        window_size: int = 100,
    ) -> None:
        self._memory_degraded = memory_degraded_pct
        self._memory_critical = memory_critical_pct
        self._latency_degraded = latency_degraded_s
        self._latency_critical = latency_critical_s
        self._error_rate_degraded = error_rate_degraded
        self._error_rate_critical = error_rate_critical

        # Rolling window for latency and error tracking
        self._window_size = window_size
        self._latencies: list[float] = []
        self._errors: list[bool] = []   # True = error, False = success

        self._mode = DegradationMode.NORMAL
        self._lock = asyncio.Lock()

    # ── Signal recording ─────────────────────────────────────────

    async def record_request(self, latency_s: float, is_error: bool) -> None:
        """Record a completed request for health monitoring."""
        async with self._lock:
            self._latencies.append(latency_s)
            if len(self._latencies) > self._window_size:
                self._latencies = self._latencies[-self._window_size:]

            self._errors.append(is_error)
            if len(self._errors) > self._window_size:
                self._errors = self._errors[-self._window_size:]

        await self._evaluate()

    # ── Mode evaluation ──────────────────────────────────────────

    async def _evaluate(self) -> None:
        """Re-evaluate degradation mode based on current signals."""
        memory_pct = psutil.virtual_memory().percent

        async with self._lock:
            avg_latency = (
                sum(self._latencies) / len(self._latencies)
                if self._latencies
                else 0.0
            )
            error_rate = (
                sum(1 for e in self._errors if e) / len(self._errors)
                if self._errors
                else 0.0
            )

        old_mode = self._mode

        # Evaluate from most severe to least
        if (
            memory_pct >= self._memory_critical
            or avg_latency >= self._latency_critical
            or error_rate >= self._error_rate_critical
        ):
            self._mode = DegradationMode.CRITICAL
        elif (
            memory_pct >= self._memory_degraded
            or avg_latency >= self._latency_degraded
            or error_rate >= self._error_rate_degraded
        ):
            self._mode = DegradationMode.DEGRADED
        else:
            self._mode = DegradationMode.NORMAL

        if self._mode != old_mode:
            logger.warning(
                "Degradation mode changed: %s -> %s "
                "(memory=%.1f%%, avg_latency=%.2fs, error_rate=%.2f)",
                old_mode.value,
                self._mode.value,
                memory_pct,
                avg_latency,
                error_rate,
            )

    # ── Public API ───────────────────────────────────────────────

    def get_mode(self) -> DegradationMode:
        """Return the current degradation mode."""
        return self._mode

    def is_feature_enabled(self, feature: str) -> bool:
        """
        Check whether a feature should be active under the current mode.

        In DEGRADED mode, search/RAG and streaming are disabled.
        In CRITICAL mode, all non-essential features are disabled.
        """
        if self._mode == DegradationMode.NORMAL:
            return True

        disabled_in_degraded = {"search_rag", "streaming"}
        disabled_in_critical = disabled_in_degraded | {"ai_queries"}

        if self._mode == DegradationMode.CRITICAL:
            return feature not in disabled_in_critical
        if self._mode == DegradationMode.DEGRADED:
            return feature not in disabled_in_degraded

        return True

    def get_health_summary(self) -> dict[str, Any]:
        """Return a health summary suitable for the health endpoint."""
        memory_pct = psutil.virtual_memory().percent
        avg_latency = (
            sum(self._latencies) / len(self._latencies)
            if self._latencies
            else 0.0
        )
        error_rate = (
            sum(1 for e in self._errors if e) / len(self._errors)
            if self._errors
            else 0.0
        )
        return {
            "mode": self._mode.value,
            "memory_usage_pct": round(memory_pct, 1),
            "avg_latency_s": round(avg_latency, 3),
            "error_rate": round(error_rate, 3),
            "sample_count": len(self._latencies),
        }
