"""
API Gateway — main FastAPI application (production-grade).

Features:
    - Global exception handler (sanitized errors)
    - Request ID generation and propagation
    - GZip response compression
    - Prometheus-compatible metrics
    - Lifecycle hooks with proper cleanup
    - OpenAPI/docs suppressed in production
    - Request size limiting
    - Search engine lifecycle
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from config import settings
from db.resilience import circuit_breaker, db_health_monitor
from db.session import dispose_engine, ensure_schema, verify_connection
from observability import Tracer, TracingMiddleware
from router import MasterRouter, IntentRouter
from search.engine import SearchEngine
from services.llm_service import GeneralLLMService
from services.voice_service import VoiceService
from cache import CacheService
from security.rate_limiter import RateLimiter

from .concurrency import AIQuerySemaphore, BackpressureMiddleware, GracefulDegradation
from .metrics import MetricsMiddleware, collector as metrics_collector, metrics_router
from .middleware import (
    AuditMiddleware,
    CorrelationIDMiddleware,
    CSRFMiddleware,
    RateLimitMiddleware,
    SecurityHeadersMiddleware,
)
from .routes import (
    auth_router, chat_router, doctor_router,
    health_router, hospital_router, patient_router,
    websocket_router,
)

logger = logging.getLogger(__name__)


# ── Lifecycle ────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hooks with proper resource cleanup."""
    logger.info(
        "Starting MedAI Platform [env=%s, version=%s]",
        settings.environment, settings.app_version,
    )

    # ── Startup ─────────────────────────────────────────────────
    # 1. Initialize General LLM service (conversational backbone)
    llm_service = GeneralLLMService()
    app.state.llm_service = llm_service

    # 2. Initialize embedding-based intent router (Layer 2+3)
    intent_router = IntentRouter()
    if not intent_router.load():
        logger.warning(
            "Embedding router not available — falling back to keyword routing. "
            "Install sentence-transformers and download BGE model to enable."
        )
    app.state.intent_router = intent_router

    # 3. Initialize search engine
    search_engine = SearchEngine()
    app.state.search_engine = search_engine

    # 4. Initialize master router with LLM backbone + intent router + search
    app.state.master_router = MasterRouter(
        llm_service=llm_service,
        intent_router=intent_router,
        search_engine=search_engine,
    )

    # 5. Initialize rate limiter (with Redis if available)
    rate_limiter = RateLimiter()
    try:
        await rate_limiter.connect_redis()
    except Exception as exc:
        logger.warning("Redis connection failed — using in-memory rate limiter: %s", exc)
    app.state.rate_limiter = rate_limiter

    # 6. Initialize cache service (shares Redis client with rate limiter)
    redis_client = getattr(rate_limiter, "_redis", None)
    app.state.redis = redis_client
    app.state.cache = CacheService(redis_client)
    logger.info(
        "Cache service initialized [redis=%s]",
        "connected" if redis_client is not None else "unavailable",
    )

    # 6b. Initialize lazy voice services (Whisper / XTTS when enabled)
    voice_service = VoiceService()
    app.state.voice_service = voice_service
    voice_capabilities = voice_service.capabilities()
    logger.info(
        "Voice service initialized [asr=%s:%s tts=%s:%s]",
        voice_capabilities.asr_provider,
        "ready" if voice_capabilities.asr_available else "disabled",
        voice_capabilities.tts_provider,
        "ready" if voice_capabilities.tts_available else "disabled",
    )

    # 7. Initialize concurrency controls
    app.state.ai_semaphore = AIQuerySemaphore(
        max_concurrent=settings.max_concurrent_ai_queries,
        queue_timeout=settings.ai_queue_timeout_seconds,
    )
    app.state.graceful_degradation = GracefulDegradation()
    logger.info(
        "Concurrency controls initialized: max_concurrent_ai=%d, max_active_requests=%d",
        settings.max_concurrent_ai_queries, settings.max_active_requests,
    )

    # 8. Initialize distributed tracer (OpenTelemetry-compatible)
    tracer = Tracer(service_name="medai-platform")
    await tracer.start()
    app.state.tracer = tracer
    logger.info("Distributed tracer initialized")

    # 9. Verify database connectivity
    if await verify_connection():
        logger.info("Database connectivity verified")
        if not settings.is_production:
            await ensure_schema()
            logger.info("Database schema synchronized for development startup")
    else:
        logger.error("Database connectivity check FAILED — platform may be degraded")

    # 10. Start database health monitor & expose circuit breaker on app.state
    app.state.circuit_breaker = circuit_breaker
    db_health_monitor.start()
    logger.info("Database health monitor started")

    logger.info("MedAI Platform ready to serve requests")

    yield

    # ── Shutdown ────────────────────────────────────────────────
    logger.info("Shutting down MedAI Platform...")

    # 0. Stop database health monitor
    db_health_monitor.stop()
    logger.info("Database health monitor stopped")

    # 1. Close engine connections
    await app.state.master_router.close()

    # 2. Close LLM service
    await app.state.llm_service.close()

    # 3. Close search engine
    await app.state.search_engine.close()

    # 4. Close rate limiter (Redis)
    await rate_limiter.close()

    # 4b. Release voice models
    await app.state.voice_service.close()

    # 5. Dispose database engine
    await dispose_engine()

    logger.info("MedAI Platform shut down cleanly")


# ── App Factory ──────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """Create the FastAPI application with all middleware and routes."""

    # Disable docs in production
    is_prod = settings.is_production
    docs_url = None if is_prod else "/docs"
    redoc_url = None if is_prod else "/redoc"
    openapi_url = None if is_prod else "/openapi.json"

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Unified Medical AI Platform — MediScan VLM + Medical LLM + OCR",
        lifespan=lifespan,
        docs_url=docs_url,
        redoc_url=redoc_url,
        openapi_url=openapi_url,
    )

    # ── Global Exception Handler ─────────────────────────────────
    @app.exception_handler(Exception)
    async def _global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """
        Catch-all error handler.
        NEVER leak stack traces or internal details to clients in production.
        """
        request_id = getattr(request.state, "request_id", "unknown")
        logger.error(
            "Unhandled exception: request_id=%s path=%s error=%s",
            request_id, request.url.path, str(exc)[:500],
            exc_info=not is_prod,  # Only include traceback in non-prod
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": "Internal server error" if is_prod else str(exc),
                "request_id": request_id,
            },
        )

    @app.exception_handler(ValueError)
    async def _value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        """Handle validation errors — only for request-validation ValueErrors."""
        import traceback as _tb

        request_id = getattr(request.state, "request_id", None)

        # Only treat as 400 if the ValueError originated from request validation
        # (our routes, auth, or schema layer). Otherwise treat as internal error.
        tb_text = "".join(_tb.format_exception(type(exc), exc, exc.__traceback__))
        _validation_markers = (
            "gateway/routes/", "gateway\\routes\\",
            "/auth/", "\\auth\\",
            "/schemas/", "\\schemas\\",
        )
        is_request_validation = any(m in tb_text for m in _validation_markers)

        if is_request_validation:
            logger.warning("ValueError in request %s: %s", request_id, exc)
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "detail": str(exc) if not is_prod else "Invalid request",
                    "request_id": request_id,
                },
            )

        # Unexpected ValueError from internal code — treat as 500
        logger.error(
            "Unexpected ValueError: request_id=%s path=%s error=%s",
            request_id, request.url.path, str(exc)[:500],
            exc_info=not is_prod,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": "Internal server error" if is_prod else str(exc),
                "request_id": request_id,
            },
        )

    # ── GZip Compression ─────────────────────────────────────────
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # ── CORS (restricted in production) ──────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=[
            "Authorization", "Content-Type", "X-Request-ID",
            "X-Correlation-ID", "X-API-Key", "X-CSRF-Token",
        ],
        expose_headers=[
            "X-RateLimit-Limit", "X-RateLimit-Remaining",
            "X-Request-ID", "X-Correlation-ID",
        ],
    )

    # ── Custom Middleware (order: outermost → innermost) ──────────
    app.add_middleware(BackpressureMiddleware, max_active_requests=settings.max_active_requests)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(CSRFMiddleware)
    app.add_middleware(CorrelationIDMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(AuditMiddleware)
    app.add_middleware(MetricsMiddleware, metrics=metrics_collector)

    # ── Store metrics collector on app state ─────────────────────
    app.state.metrics = metrics_collector

    # ── Route Registration ───────────────────────────────────────
    app.include_router(metrics_router, prefix="/api/v1", tags=["Metrics"])
    app.include_router(health_router, prefix="/api/v1", tags=["Health"])
    app.include_router(auth_router, prefix="/api/v1/auth", tags=["Authentication"])
    app.include_router(chat_router, prefix="/api/v1/chat", tags=["Chat & AI"])
    app.include_router(patient_router, prefix="/api/v1/patients", tags=["Patients"])
    app.include_router(doctor_router, prefix="/api/v1/doctors", tags=["Doctors"])
    app.include_router(hospital_router, prefix="/api/v1/hospitals", tags=["Hospitals"])

    # WebSocket routes — no /api/v1 prefix (paths defined in router as /ws/...)
    app.include_router(websocket_router, tags=["WebSocket"])

    return app
