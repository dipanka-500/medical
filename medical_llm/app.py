"""Production API for the Medical LLM Engine — hardened for market readiness."""

from __future__ import annotations

import asyncio
import logging
import logging.config
import os
import time
from contextlib import asynccontextmanager
from types import TracebackType
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, Header, HTTPException, Request, Response, status
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from main import MedicalLLMEngine

logger = logging.getLogger(__name__)


# ── Structured Logging ───────────────────────────────────────────────────

def _setup_logging() -> None:
    """Configure structured JSON logging for production."""
    log_format = os.getenv("LOG_FORMAT", "console")

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "format": '{"ts":"%(asctime)s","level":"%(levelname)s",'
                          '"logger":"%(name)s","msg":"%(message)s"}',
                "datefmt": "%Y-%m-%dT%H:%M:%S%z",
            },
            "console": {
                "format": "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
                "datefmt": "%H:%M:%S",
            },
        },
        "handlers": {
            "default": {
                "class": "logging.StreamHandler",
                "formatter": "json" if log_format == "json" else "console",
                "stream": "ext://sys.stdout",
            },
        },
        "root": {
            "level": os.getenv("LOG_LEVEL", "INFO"),
            "handlers": ["default"],
        },
        "loggers": {
            "uvicorn": {"level": "INFO", "handlers": ["default"], "propagate": False},
            "uvicorn.access": {"level": "WARNING", "handlers": ["default"], "propagate": False},
        },
    }
    logging.config.dictConfig(config)


_setup_logging()


def _env_bool(name: str, default: bool) -> bool:
    """Parse a boolean environment variable safely."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    """Parse an integer environment variable with bounds checking."""
    raw = os.getenv(name)
    if raw is None:
        return max(minimum, default)
    try:
        return max(minimum, int(raw))
    except ValueError:
        logger.warning("Invalid integer env var %s=%r; using default %s", name, raw, default)
        return max(minimum, default)


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    """Parse a float environment variable with bounds checking."""
    raw = os.getenv(name)
    if raw is None:
        return max(minimum, default)
    try:
        return max(minimum, float(raw))
    except ValueError:
        logger.warning("Invalid float env var %s=%r; using default %s", name, raw, default)
        return max(minimum, default)


class APISettings(BaseModel):
    """Runtime settings for the production API."""

    model_config_path: str = "config/model_config.yaml"
    pipeline_config_path: str = "config/pipeline_config.yaml"
    api_key: Optional[str] = None
    init_on_startup: bool = True
    ingest_builtin_knowledge: bool = False
    max_concurrent_requests: int = 1
    queue_timeout_seconds: float = 30.0
    max_queue_depth: int = 4
    redis_url: Optional[str] = None
    shared_state_enabled: bool = True
    distributed_max_concurrent_requests: int = 0
    distributed_slot_ttl_seconds: int = 360
    cache_ttl_seconds: int = 300
    session_ttl_seconds: int = 3600
    state_key_prefix: str = "medai:medical-llm"
    environment: str = "development"

    @classmethod
    def from_env(cls) -> "APISettings":
        max_concurrent_requests = _env_int("MEDICAL_LLM_MAX_CONCURRENT_REQUESTS", 1, minimum=1)
        return cls(
            model_config_path=os.getenv("MEDICAL_LLM_MODEL_CONFIG", "config/model_config.yaml"),
            pipeline_config_path=os.getenv("MEDICAL_LLM_PIPELINE_CONFIG", "config/pipeline_config.yaml"),
            api_key=os.getenv("MEDICAL_LLM_API_KEY"),
            init_on_startup=_env_bool("MEDICAL_LLM_INIT_ON_STARTUP", True),
            ingest_builtin_knowledge=_env_bool("MEDICAL_LLM_INGEST_BUILTIN", False),
            max_concurrent_requests=max_concurrent_requests,
            queue_timeout_seconds=_env_float("MEDICAL_LLM_QUEUE_TIMEOUT_SECONDS", 30.0, minimum=0.1),
            max_queue_depth=_env_int(
                "MEDICAL_LLM_MAX_QUEUE_DEPTH",
                max_concurrent_requests * 2,
                minimum=max_concurrent_requests,
            ),
            redis_url=os.getenv("MEDICAL_LLM_REDIS_URL") or os.getenv("REDIS_URL"),
            shared_state_enabled=_env_bool("MEDICAL_LLM_SHARED_STATE_ENABLED", True),
            distributed_max_concurrent_requests=_env_int(
                "MEDICAL_LLM_DISTRIBUTED_MAX_CONCURRENT_REQUESTS",
                max_concurrent_requests,
                minimum=0,
            ),
            distributed_slot_ttl_seconds=_env_int(
                "MEDICAL_LLM_DISTRIBUTED_SLOT_TTL_SECONDS", 360, minimum=30,
            ),
            cache_ttl_seconds=_env_int("MEDICAL_LLM_CACHE_TTL_SECONDS", 300, minimum=30),
            session_ttl_seconds=_env_int("MEDICAL_LLM_SESSION_TTL_SECONDS", 3600, minimum=300),
            state_key_prefix=os.getenv("MEDICAL_LLM_STATE_KEY_PREFIX", "medai:medical-llm"),
            environment=os.getenv("ENVIRONMENT", "development"),
        )


class AnalyzeRequest(BaseModel):
    """Incoming analysis request with validation."""

    query: str = Field(..., min_length=1, max_length=10000)
    mode: Literal["doctor", "patient", "research"] = "doctor"
    enable_rag: bool = True
    force_models: Optional[List[str]] = None
    use_cache: bool = True
    session_id: Optional[str] = Field(
        default=None, max_length=128,
        pattern=r"^[A-Za-z0-9_.:-]+$",
    )


# ── Metrics ──────────────────────────────────────────────────────────────

class _Metrics:
    """Thread-safe request metrics for monitoring."""

    def __init__(self) -> None:
        import threading
        self._lock = threading.Lock()
        self.total_requests: int = 0
        self.total_errors: int = 0
        self.total_latency_ms: float = 0.0

    def record(self, latency_ms: float, error: bool = False) -> None:
        with self._lock:
            self.total_requests += 1
            self.total_latency_ms += latency_ms
            if error:
                self.total_errors += 1

    def as_dict(self) -> dict:
        with self._lock:
            avg = (self.total_latency_ms / self.total_requests) if self.total_requests else 0
            return {
                "total_requests": self.total_requests,
                "total_errors": self.total_errors,
                "avg_latency_ms": round(avg, 1),
                "error_rate": round(self.total_errors / max(1, self.total_requests), 4),
            }


_DISTRIBUTED_ACQUIRE_LUA = """
local key = KEYS[1]
local limit = tonumber(ARGV[1])
local ttl = tonumber(ARGV[2])
local current = tonumber(redis.call('GET', key) or '0')
if current >= limit then
    return 0
end
current = redis.call('INCR', key)
redis.call('EXPIRE', key, ttl)
return current
"""

_DISTRIBUTED_RELEASE_LUA = """
local key = KEYS[1]
local ttl = tonumber(ARGV[1])
local current = tonumber(redis.call('GET', key) or '0')
if current <= 1 then
    redis.call('DEL', key)
    return 0
end
current = redis.call('DECR', key)
redis.call('EXPIRE', key, ttl)
return current
"""


class _RequestLease:
    """Async context manager for request slots."""

    def __init__(self, limiter: "RequestConcurrencyLimiter") -> None:
        self._limiter = limiter
        self._distributed_acquired = False

    async def __aenter__(self) -> "_RequestLease":
        self._distributed_acquired = await self._limiter.acquire()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self._limiter.release(self._distributed_acquired)


class RequestConcurrencyLimiter:
    """
    Bounds local concurrency and optionally enforces a shared Redis-wide cap.

    Local queueing protects the single process. When Redis is configured, a
    best-effort distributed in-flight counter prevents replicas from
    collectively exceeding the upstream model capacity.
    """

    _REDIS_RETRY_INTERVAL_SECONDS = 30.0

    def __init__(
        self,
        *,
        max_concurrent: int,
        queue_timeout_seconds: float,
        max_queue_depth: int,
        redis_url: str | None = None,
        distributed_max_concurrent: int = 0,
        distributed_slot_ttl_seconds: int = 360,
        key_prefix: str = "medai:medical-llm",
    ) -> None:
        self._max_concurrent = max(1, max_concurrent)
        self._queue_timeout_seconds = max(0.1, queue_timeout_seconds)
        self._max_queue_depth = max(self._max_concurrent, max_queue_depth)
        self._redis_url = redis_url
        self._distributed_max_concurrent = max(0, distributed_max_concurrent)
        self._distributed_slot_ttl_seconds = max(30, distributed_slot_ttl_seconds)
        self._distributed_key = f"{key_prefix.rstrip(':')}:inflight"

        self._local_semaphore = asyncio.Semaphore(self._max_concurrent)
        self._state_lock = asyncio.Lock()
        self._connect_lock = asyncio.Lock()
        self._active = 0
        self._waiting = 0

        self._redis: Any = None
        self._distributed_available = False
        self._last_redis_attempt = 0.0

    async def connect_redis(self) -> None:
        """Connect the optional distributed limiter backend."""
        if not self._redis_url or self._distributed_max_concurrent < 1:
            return

        try:
            import redis.asyncio as aioredis
        except Exception as exc:
            logger.warning("Redis dependency unavailable for distributed limiter: %s", exc)
            return

        try:
            redis_client = aioredis.from_url(
                self._redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_timeout=2.0,
                socket_connect_timeout=2.0,
                health_check_interval=30,
                retry_on_timeout=True,
            )
            await redis_client.ping()
        except Exception as exc:
            self._distributed_available = False
            self._redis = None
            logger.warning("Distributed request limiter unavailable: %s", exc)
            return

        self._redis = redis_client
        self._distributed_available = True
        logger.info(
            "Distributed request limiter enabled [global_max=%d]",
            self._distributed_max_concurrent,
        )

    async def close(self) -> None:
        """Close the distributed limiter backend cleanly."""
        if self._redis is None:
            return
        close = getattr(self._redis, "aclose", None)
        if callable(close):
            await close()
        else:
            await self._redis.close()
        self._redis = None
        self._distributed_available = False

    def slot(self) -> _RequestLease:
        """Return an async context manager for one request slot."""
        return _RequestLease(self)

    async def acquire(self) -> bool:
        """Acquire a request slot, shedding load when the queue is too deep."""
        await self._acquire_local_slot()
        try:
            distributed_acquired = await self._acquire_distributed_slot()
            return distributed_acquired
        except Exception:
            await self.release(False)
            raise

    async def release(self, distributed_acquired: bool) -> None:
        """Release the local slot and any distributed slot."""
        if distributed_acquired:
            await self._release_distributed_slot()
        async with self._state_lock:
            self._active = max(0, self._active - 1)
        self._local_semaphore.release()

    async def get_stats(self) -> dict[str, Any]:
        """Expose current queue and distributed limiter state."""
        global_inflight = None
        if self._distributed_available and self._redis is not None:
            try:
                raw = await self._redis.get(self._distributed_key)
                global_inflight = int(raw or 0)
            except Exception as exc:
                logger.debug("Failed to read distributed limiter stats: %s", exc)
        utilization = (self._active / self._max_concurrent) if self._max_concurrent else 0.0
        return {
            "active": self._active,
            "waiting": self._waiting,
            "max_concurrent": self._max_concurrent,
            "max_queue_depth": self._max_queue_depth,
            "queue_timeout_seconds": self._queue_timeout_seconds,
            "utilization_pct": round(utilization * 100, 1),
            "distributed_enabled": bool(self._redis_url and self._distributed_max_concurrent),
            "distributed_available": self._distributed_available,
            "distributed_max_concurrent": self._distributed_max_concurrent,
            "distributed_inflight": global_inflight,
        }

    async def _acquire_local_slot(self) -> None:
        async with self._state_lock:
            if self._waiting >= self._max_queue_depth:
                raise OverflowError(
                    f"AI request queue is full ({self._waiting}/{self._max_queue_depth}).",
                )
            self._waiting += 1

        try:
            await asyncio.wait_for(
                self._local_semaphore.acquire(),
                timeout=self._queue_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            async with self._state_lock:
                self._waiting = max(0, self._waiting - 1)
            raise TimeoutError(
                f"Timed out waiting for an AI request slot after "
                f"{self._queue_timeout_seconds:.1f}s.",
            ) from exc

        async with self._state_lock:
            self._waiting = max(0, self._waiting - 1)
            self._active += 1

    async def _ensure_distributed_backend(self) -> None:
        if self._distributed_available or not self._redis_url or self._distributed_max_concurrent < 1:
            return
        now = time.monotonic()
        if now - self._last_redis_attempt < self._REDIS_RETRY_INTERVAL_SECONDS:
            return
        async with self._connect_lock:
            now = time.monotonic()
            if now - self._last_redis_attempt < self._REDIS_RETRY_INTERVAL_SECONDS:
                return
            self._last_redis_attempt = now
            await self.connect_redis()

    async def _acquire_distributed_slot(self) -> bool:
        await self._ensure_distributed_backend()
        if not self._distributed_available or self._redis is None:
            return False
        try:
            result = await self._redis.eval(
                _DISTRIBUTED_ACQUIRE_LUA,
                1,
                self._distributed_key,
                self._distributed_max_concurrent,
                self._distributed_slot_ttl_seconds,
            )
        except Exception as exc:
            self._distributed_available = False
            logger.warning("Distributed limiter degraded to local-only mode: %s", exc)
            return False

        if int(result or 0) <= 0:
            raise OverflowError("Global AI request capacity has been reached.")
        return True

    async def _release_distributed_slot(self) -> None:
        if not self._distributed_available or self._redis is None:
            return
        try:
            await self._redis.eval(
                _DISTRIBUTED_RELEASE_LUA,
                1,
                self._distributed_key,
                self._distributed_slot_ttl_seconds,
            )
        except Exception as exc:
            self._distributed_available = False
            logger.warning("Failed to release distributed request slot cleanly: %s", exc)


def create_app(api_settings: Optional[APISettings] = None) -> FastAPI:
    """Create the FastAPI application."""
    api_settings = api_settings or APISettings.from_env()
    is_prod = api_settings.environment == "production"
    metrics = _Metrics()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Handle startup/shutdown."""
        logger.info("Medical LLM Engine starting (env=%s)", api_settings.environment)
        await app.state.request_limiter.connect_redis()
        if api_settings.init_on_startup:
            try:
                await ensure_engine_initialized()
                logger.info("Engine initialized successfully on startup")
            except Exception:
                logger.exception("Startup initialization failed")
        yield
        # Graceful shutdown: wait for in-flight requests
        logger.info("Shutting down Medical LLM Engine (draining requests)...")
        await app.state.request_limiter.close()

    app = FastAPI(
        title="Medical LLM Engine API",
        version="1.0.0",
        docs_url=None if is_prod else "/docs",
        redoc_url=None if is_prod else "/redoc",
        openapi_url=None if is_prod else "/openapi.json",
        lifespan=lifespan,
    )

    # GZip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    app.state.settings = api_settings
    app.state.engine = MedicalLLMEngine(
        model_config_path=api_settings.model_config_path,
        pipeline_config_path=api_settings.pipeline_config_path,
        redis_url=api_settings.redis_url,
        enable_shared_state=api_settings.shared_state_enabled,
        cache_ttl_seconds=api_settings.cache_ttl_seconds,
        session_ttl_seconds=api_settings.session_ttl_seconds,
        shared_state_prefix=api_settings.state_key_prefix,
    )
    app.state.request_limiter = RequestConcurrencyLimiter(
        max_concurrent=api_settings.max_concurrent_requests,
        queue_timeout_seconds=api_settings.queue_timeout_seconds,
        max_queue_depth=api_settings.max_queue_depth,
        redis_url=api_settings.redis_url,
        distributed_max_concurrent=api_settings.distributed_max_concurrent_requests,
        distributed_slot_ttl_seconds=api_settings.distributed_slot_ttl_seconds,
        key_prefix=api_settings.state_key_prefix,
    )
    app.state.init_lock = asyncio.Lock()
    app.state.startup_error = None
    app.state.builtin_ingested = False
    app.state.metrics = metrics

    # ── Global exception handler ─────────────────────────────────
    @app.exception_handler(Exception)
    async def _global_error(request: Request, exc: Exception) -> Response:
        logger.error("Unhandled error: %s", exc, exc_info=not is_prod)
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error" if is_prod else str(exc)},
        )

    async def ensure_engine_initialized() -> None:
        """Initialize the engine once, with async-safe locking."""
        if app.state.engine._is_initialized:
            return
        async with app.state.init_lock:
            if app.state.engine._is_initialized:
                return
            def _init_sync() -> None:
                app.state.engine.initialize()
                if api_settings.ingest_builtin_knowledge and not app.state.builtin_ingested:
                    app.state.engine.ingest_built_in_knowledge()
                    app.state.builtin_ingested = True
            try:
                await run_in_threadpool(_init_sync)
                app.state.startup_error = None
            except Exception as exc:
                app.state.startup_error = str(exc)
                logger.exception("Engine initialization failed")
                raise exc

    async def require_api_key(
        x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    ) -> None:
        expected_key = app.state.settings.api_key
        if not expected_key:
            return
        import hmac
        if not x_api_key or not hmac.compare_digest(x_api_key, expected_key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
            )

    # ── Endpoints ────────────────────────────────────────────────

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        """Liveness probe."""
        return {
            "status": "ok",
            "service": "medical-llm-api",
            "engine_initialized": app.state.engine._is_initialized,
        }

    @app.get("/ready")
    async def ready(response: Response) -> Dict[str, Any]:
        """Readiness probe."""
        startup_error = app.state.startup_error
        initialized = app.state.engine._is_initialized

        if startup_error:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return {"status": "error", "ready": False, "error": startup_error}

        if not initialized:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return {"status": "initializing", "ready": False}

        state_backend = (
            app.state.engine.get_state_backend_status()
            if hasattr(app.state.engine, "get_state_backend_status")
            else {"mode": "memory"}
        )
        return {
            "status": "ready",
            "ready": True,
            "loaded_models": len(app.state.engine.get_model_status()),
            "conversation_sessions": app.state.engine.get_conversation_session_count(),
            "state_backend": state_backend,
        }

    @app.get("/status")
    async def runtime_status(request: Request) -> Dict[str, Any]:
        """Runtime status for operators."""
        await require_api_key(request.headers.get("X-API-Key"))
        initialized = app.state.engine._is_initialized
        state_backend = (
            app.state.engine.get_state_backend_status()
            if hasattr(app.state.engine, "get_state_backend_status")
            else {"mode": "memory"}
        )
        return {
            "initialized": initialized,
            "startup_error": app.state.startup_error,
            "max_concurrent_requests": api_settings.max_concurrent_requests,
            "conversation_sessions": app.state.engine.get_conversation_session_count(),
            "models": app.state.engine.get_model_status() if initialized else {},
            "metrics": metrics.as_dict(),
            "state_backend": state_backend,
            "request_limiter": await app.state.request_limiter.get_stats(),
        }

    @app.post("/analyze")
    async def analyze(
        request: AnalyzeRequest,
        x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    ) -> Dict[str, Any]:
        """Run the engine for a single analysis request."""
        await require_api_key(x_api_key)
        await ensure_engine_initialized()

        started = time.perf_counter()
        try:
            async with app.state.request_limiter.slot():
                result = await asyncio.wait_for(
                    run_in_threadpool(
                        app.state.engine.analyze,
                        request.query,
                        request.mode,
                        request.enable_rag,
                        request.force_models,
                        request.use_cache,
                        request.session_id,
                    ),
                    timeout=300,
                )
                latency_ms = (time.perf_counter() - started) * 1000
                result["api_latency_seconds"] = round(latency_ms / 1000, 3)
                metrics.record(latency_ms)
                return result
        except OverflowError as exc:
            latency_ms = (time.perf_counter() - started) * 1000
            metrics.record(latency_ms, error=True)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
                headers={"Retry-After": "5"},
            ) from exc
        except TimeoutError as exc:
            latency_ms = (time.perf_counter() - started) * 1000
            metrics.record(latency_ms, error=True)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
                headers={"Retry-After": "5"},
            ) from exc
        except asyncio.TimeoutError:
            latency_ms = (time.perf_counter() - started) * 1000
            metrics.record(latency_ms, error=True)
            logger.error("Analysis timed out after 300s")
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Analysis timed out",
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - started) * 1000
            metrics.record(latency_ms, error=True)
            logger.error("Analysis failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Analysis failed" if is_prod else str(exc),
            )

    @app.delete("/sessions/{session_id}")
    async def clear_session(
        session_id: str,
        x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    ) -> Dict[str, Any]:
        """Delete stored conversation history for a session."""
        await require_api_key(x_api_key)
        cleared = app.state.engine.clear_conversation(session_id)
        return {"session_id": session_id, "cleared": cleared}

    @app.get("/metrics")
    async def get_metrics(
        x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    ) -> Dict[str, Any]:
        """Prometheus-compatible metrics."""
        await require_api_key(x_api_key)
        state_backend = (
            app.state.engine.get_state_backend_status()
            if hasattr(app.state.engine, "get_state_backend_status")
            else {"mode": "memory"}
        )
        return {
            "service": "medical-llm-api",
            "request_metrics": metrics.as_dict(),
            "engine_initialized": app.state.engine._is_initialized,
            "request_limiter": await app.state.request_limiter.get_stats(),
            "state_backend": state_backend,
        }

    # ── RAG Endpoints (consumed by platform search engine + master router) ──

    @app.post("/rag/search")
    async def rag_search(
        request: Request,
        x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    ) -> Dict[str, Any]:
        """
        Search the knowledge base for relevant medical information.
        Used by platform's SearchEngine and MasterRouter.
        """
        await require_api_key(x_api_key)
        await ensure_engine_initialized()

        body = await request.json()
        query = body.get("query", "")
        max_results = min(body.get("max_results", 10), 50)
        sources = body.get("sources")

        if not query.strip():
            return {"sources": [], "total": 0}

        try:
            async with app.state.request_limiter.slot():
                # Use the engine's RAG capabilities
                engine = app.state.engine
                if hasattr(engine, "search_knowledge_base"):
                    results = await run_in_threadpool(
                        engine.search_knowledge_base, query, max_results, sources,
                    )
                    if isinstance(results, dict):
                        return results
                elif hasattr(engine, "rag_search"):
                    results = await run_in_threadpool(
                        engine.rag_search, query, max_results,
                    )
                else:
                    # Fallback: run a basic analyze in research mode to get sources
                    result = await run_in_threadpool(
                        engine.analyze, query, "research", True, None, True, None,
                    )
                    return {
                        "sources": result.get("sources", []),
                        "total": len(result.get("sources", [])),
                    }

                return {
                    "sources": results if isinstance(results, list) else [],
                    "total": len(results) if isinstance(results, list) else 0,
                }
        except (OverflowError, TimeoutError) as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
                headers={"Retry-After": "5"},
            ) from exc
        except Exception as exc:
            logger.error("RAG search failed: %s", exc)
            return {"sources": [], "total": 0, "error": "Internal search error"}

    @app.post("/rag/query")
    async def rag_query(
        request: Request,
        x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    ) -> Dict[str, Any]:
        """
        Query the vector database for semantically similar content.
        Used by platform's SearchEngine for vector_db source.
        """
        await require_api_key(x_api_key)
        await ensure_engine_initialized()

        body = await request.json()
        query = body.get("query", "")
        top_k = min(body.get("top_k", 10), 50)

        if not query.strip():
            return {"results": [], "total": 0}

        try:
            async with app.state.request_limiter.slot():
                engine = app.state.engine
                if hasattr(engine, "query_vector_db"):
                    results = await run_in_threadpool(
                        engine.query_vector_db, query, top_k,
                    )
                    return {
                        "results": results if isinstance(results, list) else [],
                        "total": len(results) if isinstance(results, list) else 0,
                    }
                elif hasattr(engine, "rag_query"):
                    results = await run_in_threadpool(
                        engine.rag_query, query, top_k,
                    )
                else:
                    # Fallback: use analyze to simulate vector search
                    result = await run_in_threadpool(
                        engine.analyze, query, "research", True, None, True, None,
                    )
                    sources = result.get("sources", [])
                    return {
                        "results": [
                            {
                                "content": s.get("content", ""),
                                "title": s.get("title", "Knowledge Base"),
                                "score": s.get("relevance_score", 0.7),
                            }
                            for s in sources[:top_k]
                        ],
                        "total": len(sources),
                    }

                return {
                    "results": results if isinstance(results, list) else [],
                    "total": len(results) if isinstance(results, list) else 0,
                }
        except (OverflowError, TimeoutError) as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
                headers={"Retry-After": "5"},
            ) from exc
        except Exception as exc:
            logger.error("RAG query failed: %s", exc)
            return {"results": [], "total": 0, "error": "Internal query error"}

    return app


app = create_app()
