"""
Lightweight distributed tracing — OpenTelemetry-compatible data model.

Provides W3C Trace Context (traceparent) propagation, span lifecycle
management, and stdout JSON export. Designed as a drop-in that can be
swapped for real OpenTelemetry SDK later by replacing this module.

Uses only the Python standard library (no external packages).
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request, Response

logger = logging.getLogger(__name__)

# ── Span Data Model (OpenTelemetry-compatible) ───────────────────────────


def _new_trace_id() -> str:
    """Generate a 32-char lowercase hex trace ID."""
    return uuid.uuid4().hex


def _new_span_id() -> str:
    """Generate a 16-char lowercase hex span ID."""
    return uuid.uuid4().hex[:16]


@dataclass
class TraceContext:
    """
    Represents a single span in a distributed trace.

    Field semantics match OpenTelemetry's Span data model so migration
    to the real SDK only requires changing the exporter, not call sites.
    """

    trace_id: str
    span_id: str
    parent_span_id: str | None
    service_name: str
    operation_name: str
    start_time: float
    end_time: float | None = None
    status: str = "OK"  # OK | ERROR
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)

    # ── Convenience helpers ──────────────────────────────────────

    @property
    def duration_ms(self) -> float | None:
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        self.events.append({
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {},
        })

    def set_error(self, exc: BaseException) -> None:
        self.status = "ERROR"
        self.attributes["error.type"] = type(exc).__name__
        self.attributes["error.message"] = str(exc)[:500]
        self.add_event("exception", {
            "exception.type": type(exc).__name__,
            "exception.message": str(exc)[:500],
        })


# ── Span Exporter (stdout JSON lines) ────────────────────────────────────


class SpanExporter:
    """
    Batches completed spans and flushes them as JSON lines to stdout.

    Flush triggers:
        - Every ``flush_interval`` seconds (background task)
        - When the buffer reaches ``max_batch_size`` spans
    """

    def __init__(
        self,
        flush_interval: float = 5.0,
        max_batch_size: int = 100,
        output_stream: Any = None,
    ) -> None:
        self._buffer: list[dict[str, Any]] = []
        self._lock = asyncio.Lock()
        self._flush_interval = flush_interval
        self._max_batch_size = max_batch_size
        self._output = output_stream or sys.stdout
        self._flush_task: asyncio.Task[None] | None = None

    # ── Public API ───────────────────────────────────────────────

    async def export(self, span: TraceContext) -> None:
        """Add a completed span to the buffer; auto-flush if full."""
        record = {
            "trace_id": span.trace_id,
            "span_id": span.span_id,
            "parent_span_id": span.parent_span_id,
            "service": span.service_name,
            "operation": span.operation_name,
            "start_time": span.start_time,
            "end_time": span.end_time,
            "duration_ms": round(span.duration_ms, 3) if span.duration_ms is not None else None,
            "status": span.status,
            "attributes": span.attributes,
            "events": span.events,
        }
        async with self._lock:
            self._buffer.append(record)
            if len(self._buffer) >= self._max_batch_size:
                await self._flush_locked()

    async def flush(self) -> None:
        async with self._lock:
            await self._flush_locked()

    async def start(self) -> None:
        """Start the background flush loop."""
        if self._flush_task is None:
            self._flush_task = asyncio.create_task(self._flush_loop())

    async def stop(self) -> None:
        """Cancel the background loop and do a final flush."""
        if self._flush_task is not None:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
            self._flush_task = None
        await self.flush()

    # ── Internal ─────────────────────────────────────────────────

    async def _flush_locked(self) -> None:
        """Write buffered spans as JSON lines. Caller must hold ``_lock``."""
        if not self._buffer:
            return
        batch = self._buffer[:]
        self._buffer.clear()
        for record in batch:
            try:
                line = json.dumps(record, default=str)
                self._output.write(line + "\n")
                self._output.flush()
            except Exception:
                logger.debug("Failed to export span", exc_info=True)

    async def _flush_loop(self) -> None:
        """Background loop — flushes at regular intervals."""
        while True:
            await asyncio.sleep(self._flush_interval)
            await self.flush()


# ── Tracer ───────────────────────────────────────────────────────────────


class Tracer:
    """
    Lightweight tracer that creates spans, manages context, and exports
    completed spans via ``SpanExporter``.
    """

    def __init__(
        self,
        service_name: str = "medai-platform",
        exporter: SpanExporter | None = None,
    ) -> None:
        self.service_name = service_name
        self.exporter = exporter or SpanExporter()

    # ── Span Lifecycle ───────────────────────────────────────────

    def start_span(
        self,
        operation_name: str,
        parent: TraceContext | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> TraceContext:
        """Create and return a new span (optionally as a child of *parent*)."""
        return TraceContext(
            trace_id=parent.trace_id if parent else _new_trace_id(),
            span_id=_new_span_id(),
            parent_span_id=parent.span_id if parent else None,
            service_name=self.service_name,
            operation_name=operation_name,
            start_time=time.time(),
            attributes=dict(attributes) if attributes else {},
        )

    async def end_span(
        self,
        span: TraceContext,
        status: str = "OK",
        error: BaseException | None = None,
    ) -> None:
        """Finalize and export the span."""
        span.end_time = time.time()
        if error is not None:
            span.set_error(error)
        else:
            span.status = status
        await self.exporter.export(span)

    # ── W3C Trace Context Propagation ────────────────────────────

    @staticmethod
    def inject_headers(span: TraceContext) -> dict[str, str]:
        """
        Build W3C ``traceparent`` header for outbound requests.

        Format: ``{version}-{trace_id}-{span_id}-{flags}``
        """
        traceparent = f"00-{span.trace_id}-{span.span_id}-01"
        return {"traceparent": traceparent}

    def extract_from_headers(self, headers: dict[str, str] | Any) -> TraceContext | None:
        """
        Parse an incoming ``traceparent`` header and return a context
        that can be used as a parent span.

        Returns ``None`` if the header is missing or malformed.
        """
        raw = None
        if hasattr(headers, "get"):
            raw = headers.get("traceparent")
        if not raw:
            return None
        parts = raw.strip().split("-")
        if len(parts) < 4:
            return None
        try:
            _version, trace_id, parent_span_id, _flags = parts[:4]
            if len(trace_id) != 32 or len(parent_span_id) != 16:
                return None
            # Validate hex
            int(trace_id, 16)
            int(parent_span_id, 16)
        except (ValueError, TypeError):
            return None

        return TraceContext(
            trace_id=trace_id,
            span_id=parent_span_id,
            parent_span_id=None,
            service_name=self.service_name,
            operation_name="incoming",
            start_time=time.time(),
        )

    # ── Lifecycle helpers (call from app lifespan) ───────────────

    async def start(self) -> None:
        await self.exporter.start()

    async def stop(self) -> None:
        await self.exporter.stop()


# ── Tracing Middleware (Starlette / FastAPI) ──────────────────────────────

_SKIP_TRACING_PATHS = frozenset({
    "/health", "/api/v1/health", "/api/v1/health/ready",
    "/api/v1/health/live", "/docs", "/openapi.json", "/redoc",
    "/favicon.ico",
})


class TracingMiddleware(BaseHTTPMiddleware):
    """
    HTTP middleware that creates a root span per request and propagates
    the W3C ``traceparent`` header on responses.
    """

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        # Skip health-check and documentation endpoints
        if request.url.path in _SKIP_TRACING_PATHS:
            return await call_next(request)

        tracer: Tracer | None = getattr(request.app.state, "tracer", None)
        if tracer is None:
            return await call_next(request)

        # Extract parent context from incoming headers (if present)
        parent_ctx = tracer.extract_from_headers(request.headers)

        # Create a span for this HTTP request
        span = tracer.start_span(
            operation_name=f"HTTP {request.method} {request.url.path}",
            parent=parent_ctx,
            attributes={
                "http.method": request.method,
                "http.url": str(request.url),
                "http.route": request.url.path,
                "http.client_ip": request.client.host if request.client else "unknown",
            },
        )

        # Store on request.state so downstream code can create child spans
        request.state.trace_context = span

        error: BaseException | None = None
        response: Response | None = None
        try:
            response = await call_next(request)
            span.attributes["http.status_code"] = response.status_code
            return response
        except Exception as exc:
            error = exc
            span.attributes["http.status_code"] = 500
            raise
        finally:
            status = "ERROR" if error or (response and response.status_code >= 500) else "OK"
            await tracer.end_span(span, status=status, error=error)

            # Add traceparent to response for downstream propagation
            if response is not None:
                traceparent = f"00-{span.trace_id}-{span.span_id}-01"
                response.headers["traceparent"] = traceparent


# ── @trace Decorator ─────────────────────────────────────────────────────


def trace(operation_name: str) -> Callable:
    """
    Decorator for **async** functions that automatically creates a child span.

    Usage::

        @trace("db.query")
        async def run_query(session, sql, **kwargs):
            ...

    The decorator inspects the call's keyword/positional arguments for a
    ``request`` object (FastAPI ``Request``) to find the current trace
    context.  If none is found, the span is created as a root span.
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # --- Locate tracer and parent context ---
            tracer: Tracer | None = None
            parent: TraceContext | None = None

            # Search args/kwargs for a Request object
            sig = inspect.signature(fn)
            bound = None
            try:
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
            except TypeError:
                pass

            all_vals = list(kwargs.values()) + list(args)
            if bound:
                all_vals = list(bound.arguments.values())

            for val in all_vals:
                if isinstance(val, Request):
                    tracer = getattr(val.app.state, "tracer", None)
                    parent = getattr(val.state, "trace_context", None)
                    break

            # If no tracer found, just run the function normally
            if tracer is None:
                return await fn(*args, **kwargs)

            # Build safe attributes from function arguments
            safe_attrs: dict[str, Any] = {"function": fn.__qualname__}
            if bound:
                for param_name, param_val in bound.arguments.items():
                    if param_name in ("self", "cls", "request", "session", "db"):
                        continue
                    try:
                        safe_attrs[f"arg.{param_name}"] = str(param_val)[:200]
                    except Exception:
                        pass

            span = tracer.start_span(
                operation_name=operation_name,
                parent=parent,
                attributes=safe_attrs,
            )

            try:
                result = await fn(*args, **kwargs)
                await tracer.end_span(span, status="OK")
                return result
            except Exception as exc:
                await tracer.end_span(span, status="ERROR", error=exc)
                raise

        return wrapper

    return decorator
