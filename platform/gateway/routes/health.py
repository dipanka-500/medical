"""
Health & system status endpoints — production-grade.

K8s-compatible probes:
    /health/live  — Liveness (process is alive)
    /health/ready — Readiness (all dependencies up)
    /health       — Deep health (engine + DB + Redis status)

Features:
    - Database connectivity check
    - Redis connectivity check
    - Engine health with circuit breaker status
    - Pool status monitoring
    - System resource reporting
    - Degraded mode support
"""

from __future__ import annotations

import logging
import os
import platform as stdlib_platform
import time
from typing import Any

from fastapi import APIRouter, Request, Response, status

from config import settings
from db.session import get_pool_status, verify_connection

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Liveness Probe ───────────────────────────────────────────────────────

@router.get("/health/live")
async def liveness_check():
    """
    K8s liveness probe — returns 200 if the process is alive.
    If this fails, K8s should restart the pod.
    Always returns 200 (if the process can serve requests, it's alive).
    """
    return {"status": "alive", "service": "medai-platform"}


# ── Readiness Probe ──────────────────────────────────────────────────────

@router.get("/health/ready")
async def readiness_check(request: Request, response: Response):
    """
    K8s readiness probe — returns 200 if ALL dependencies are healthy.
    If this fails, K8s removes the pod from the service load balancer.
    """
    checks: dict[str, bool] = {}

    # 1. Database check (circuit breaker + connectivity)
    cb = getattr(request.app.state, "circuit_breaker", None)
    if cb is not None and cb.state.value == "OPEN":
        # Circuit is OPEN — database is known-unhealthy, skip the probe
        checks["database"] = False
        checks["database_circuit_breaker"] = cb.get_state()
    else:
        try:
            checks["database"] = await verify_connection()
        except Exception:
            checks["database"] = False
        if cb is not None:
            checks["database_circuit_breaker"] = cb.get_state()

    # 2. Redis check (via rate limiter)
    redis_client = getattr(request.app.state, "redis", None)
    if redis_client is None:
        rate_limiter = getattr(request.app.state, "rate_limiter", None)
        redis_client = getattr(rate_limiter, "_redis", None) if rate_limiter else None
    if redis_client is not None:
        try:
            checks["redis"] = await redis_client.ping()
        except Exception:
            checks["redis"] = False
    else:
        checks["redis"] = True  # No Redis configured — in-memory fallback

    all_ready = all(checks.values())

    if not all_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return {
        "ready": all_ready,
        "checks": checks,
    }


# ── Deep Health Check ────────────────────────────────────────────────────

@router.get("/health")
async def deep_health_check(request: Request, response: Response):
    """
    Comprehensive health check — checks ALL subsystems.

    Returns:
        - Overall status: healthy / degraded / unhealthy
        - Individual engine health with circuit breaker state
        - Database pool status
        - Platform metadata
    """
    start = time.monotonic()
    health: dict[str, Any] = {
        "status": "healthy",
        "version": settings.app_version,
        "environment": settings.environment,
    }

    issues: list[str] = []
    critical_up = True

    # ── 1. Database ──────────────────────────────────────────────
    cb = getattr(request.app.state, "circuit_breaker", None)
    cb_state = cb.get_state() if cb is not None else None

    if cb is not None and cb.state.value == "OPEN":
        # Circuit is OPEN — skip the DB probe entirely
        health["database"] = {
            "status": "unhealthy",
            "circuit_breaker": cb_state,
            "pool": get_pool_status(),
        }
        issues.append("Database circuit breaker is OPEN — connections rejected")
        critical_up = False
    else:
        try:
            db_ok = await verify_connection()
            pool_status = get_pool_status()
            health["database"] = {
                "status": "healthy" if db_ok else "unhealthy",
                "pool": pool_status,
            }
            if cb_state is not None:
                health["database"]["circuit_breaker"] = cb_state
            if not db_ok:
                issues.append("Database connectivity failed")
                critical_up = False

            # Report pool metrics to Prometheus collector if available
            metrics_collector = getattr(request.app.state, "metrics", None)
            if metrics_collector and isinstance(pool_status, dict):
                for source_key, state_name in (
                    ("checked_out", "active"),
                    ("checked_in", "idle"),
                    ("overflow", "overflow"),
                ):
                    value = pool_status.get(source_key)
                    if value is not None:
                        metrics_collector.set_gauge(
                            "medai_db_pool_connections",
                            float(value),
                            {"state": state_name},
                        )
        except Exception as exc:
            health["database"] = {"status": "unhealthy", "error": str(exc)[:100]}
            if cb_state is not None:
                health["database"]["circuit_breaker"] = cb_state
            issues.append(f"Database error: {str(exc)[:100]}")
            critical_up = False

    # ── 2. AI Engines ────────────────────────────────────────────
    master_router = getattr(request.app.state, "master_router", None)
    if master_router:
        try:
            engine_health = await master_router.health_check()
            health["engines"] = engine_health

            for name, info in engine_health.items():
                if info.get("status") != "healthy":
                    issues.append(f"Engine {name}: {info.get('status', 'unknown')}")
        except Exception as exc:
            health["engines"] = {"error": str(exc)[:100]}
            issues.append(f"Engine health check error: {str(exc)[:100]}")
    else:
        health["engines"] = {"status": "not_initialized"}

    # ── 3. Rate Limiter ──────────────────────────────────────────
    rate_limiter = getattr(request.app.state, "rate_limiter", None)
    if rate_limiter:
        health["rate_limiter"] = await rate_limiter.get_metrics()
    else:
        health["rate_limiter"] = {"status": "not_initialized"}

    # ── 4. System Resources ──────────────────────────────────────
    try:
        health["system"] = {
            "python_version": stdlib_platform.python_version(),
            "platform": stdlib_platform.platform(),
            "pid": os.getpid(),
        }
    except Exception:
        pass

    # ── Determine overall status ─────────────────────────────────
    latency_ms = (time.monotonic() - start) * 1000
    health["health_check_latency_ms"] = round(latency_ms, 1)

    if not critical_up:
        health["status"] = "unhealthy"
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    elif issues:
        health["status"] = "degraded"

    if issues:
        health["issues"] = issues

    return health


# ── Metrics Endpoint (Prometheus-compatible) ─────────────────────────────

@router.get("/health/metrics")
async def metrics(request: Request):
    """
    Prometheus-compatible metrics endpoint.

    Exposes:
        - Rate limiter stats
        - DB pool stats
        - Platform metadata
    """
    metrics_data: dict[str, Any] = {
        "service": "medai-platform",
        "version": settings.app_version,
        "environment": settings.environment,
    }

    # DB pool metrics
    try:
        metrics_data["db_pool"] = get_pool_status()
    except Exception:
        metrics_data["db_pool"] = {}

    # Rate limiter metrics
    rate_limiter = getattr(request.app.state, "rate_limiter", None)
    if rate_limiter:
        metrics_data["rate_limiter"] = await rate_limiter.get_metrics()

    return metrics_data
