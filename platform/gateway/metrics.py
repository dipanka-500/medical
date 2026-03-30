"""
Prometheus-compatible metrics system for MedAI Platform.

Self-contained implementation — no dependency on prometheus_client.
Uses dict-based storage with threading.Lock for thread safety.
Exports metrics in Prometheus text exposition format (text/plain).

Metric names follow Prometheus naming conventions:
    - medai_http_requests_total
    - medai_http_request_duration_seconds
    - medai_active_requests
    - medai_ai_queries_total
    - medai_ai_query_duration_seconds
    - medai_db_pool_connections
    - medai_rate_limit_rejections_total
    - medai_auth_events_total
    - medai_errors_total
"""

from __future__ import annotations

import math
import re
import time
import threading
from typing import Any

from fastapi import APIRouter, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

# ── Histogram bucket boundaries (seconds) ───────────────────────────────
# Follows Prometheus default + extra sub-100ms buckets for fast APIs.
DEFAULT_BUCKETS = (
    0.005, 0.01, 0.025, 0.05, 0.075,
    0.1, 0.25, 0.5, 0.75,
    1.0, 2.5, 5.0, 7.5, 10.0,
    float("inf"),
)

# UUID regex for path normalisation
_UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)
# Numeric ID segments (e.g. /patients/42)
_NUMERIC_ID_RE = re.compile(r"/\d+(?=/|$)")


def _normalise_path(path: str) -> str:
    """Replace UUIDs and numeric IDs in URL paths with ``{id}`` to prevent
    cardinality explosion in metric labels."""
    path = _UUID_RE.sub("{id}", path)
    path = _NUMERIC_ID_RE.sub("/{id}", path)
    return path


# ═══════════════════════════════════════════════════════════════════════════
# Metrics Collector
# ═══════════════════════════════════════════════════════════════════════════

class MetricsCollector:
    """Thread-safe, dict-backed metrics collector that exports Prometheus
    text exposition format without any external dependency."""

    def __init__(self, *, histogram_buckets: tuple[float, ...] = DEFAULT_BUCKETS):
        self._lock = threading.Lock()
        self._buckets = histogram_buckets

        # ── Counters  {metric_name: {labels_key: float}}
        self._counters: dict[str, dict[str, float]] = {}
        # ── Gauges    {metric_name: {labels_key: float}}
        self._gauges: dict[str, dict[str, float]] = {}
        # ── Histograms {metric_name: {labels_key: {"buckets": [...], "sum": float, "count": int}}}
        self._histograms: dict[str, dict[str, dict[str, Any]]] = {}

        # ── Help / Type metadata for text output
        self._meta: dict[str, tuple[str, str]] = {}  # name -> (type, help)

        # Pre-register all MedAI metrics
        self._register_metrics()

    # ── Registration ─────────────────────────────────────────────────────

    def _register_metrics(self) -> None:
        self._register_counter(
            "medai_http_requests_total",
            "Total HTTP requests processed.",
        )
        self._register_histogram(
            "medai_http_request_duration_seconds",
            "HTTP request duration in seconds.",
        )
        self._register_gauge(
            "medai_active_requests",
            "Number of currently in-flight HTTP requests.",
        )
        self._register_counter(
            "medai_ai_queries_total",
            "Total AI queries processed.",
        )
        self._register_histogram(
            "medai_ai_query_duration_seconds",
            "AI query latency in seconds.",
        )
        self._register_gauge(
            "medai_db_pool_connections",
            "Database connection pool connections by state.",
        )
        self._register_counter(
            "medai_rate_limit_rejections_total",
            "Total requests rejected by rate limiting.",
        )
        self._register_counter(
            "medai_auth_events_total",
            "Total authentication events.",
        )
        self._register_counter(
            "medai_errors_total",
            "Total errors encountered.",
        )

    def _register_counter(self, name: str, help_text: str) -> None:
        self._meta[name] = ("counter", help_text)
        self._counters.setdefault(name, {})

    def _register_gauge(self, name: str, help_text: str) -> None:
        self._meta[name] = ("gauge", help_text)
        self._gauges.setdefault(name, {})

    def _register_histogram(self, name: str, help_text: str) -> None:
        self._meta[name] = ("histogram", help_text)
        self._histograms.setdefault(name, {})

    # ── Public API — mutators ────────────────────────────────────────────

    @staticmethod
    def _labels_key(labels: dict[str, str]) -> str:
        """Convert a label dict into a canonical, hashable string."""
        if not labels:
            return ""
        parts = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{{{parts}}}"

    def inc_counter(self, name: str, labels: dict[str, str] | None = None, value: float = 1.0) -> None:
        key = self._labels_key(labels or {})
        with self._lock:
            bucket = self._counters.setdefault(name, {})
            bucket[key] = bucket.get(key, 0.0) + value

    def set_gauge(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        key = self._labels_key(labels or {})
        with self._lock:
            bucket = self._gauges.setdefault(name, {})
            bucket[key] = value

    def inc_gauge(self, name: str, labels: dict[str, str] | None = None, value: float = 1.0) -> None:
        key = self._labels_key(labels or {})
        with self._lock:
            bucket = self._gauges.setdefault(name, {})
            bucket[key] = bucket.get(key, 0.0) + value

    def dec_gauge(self, name: str, labels: dict[str, str] | None = None, value: float = 1.0) -> None:
        key = self._labels_key(labels or {})
        with self._lock:
            bucket = self._gauges.setdefault(name, {})
            bucket[key] = bucket.get(key, 0.0) - value

    def observe_histogram(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        key = self._labels_key(labels or {})
        with self._lock:
            series = self._histograms.setdefault(name, {})
            if key not in series:
                series[key] = {
                    "buckets": [0] * len(self._buckets),
                    "sum": 0.0,
                    "count": 0,
                }
            entry = series[key]
            entry["sum"] += value
            entry["count"] += 1
            for i, bound in enumerate(self._buckets):
                if value <= bound:
                    entry["buckets"][i] += 1

    # ── Exposition ───────────────────────────────────────────────────────

    def export(self) -> str:
        """Return all metrics in Prometheus text exposition format."""
        lines: list[str] = []

        with self._lock:
            # Counters
            for name, series in sorted(self._counters.items()):
                mtype, mhelp = self._meta.get(name, ("counter", ""))
                lines.append(f"# HELP {name} {mhelp}")
                lines.append(f"# TYPE {name} {mtype}")
                if not series:
                    lines.append(f"{name} 0")
                else:
                    for lk, val in sorted(series.items()):
                        lines.append(f"{name}{lk} {self._fmt(val)}")

            # Gauges
            for name, series in sorted(self._gauges.items()):
                mtype, mhelp = self._meta.get(name, ("gauge", ""))
                lines.append(f"# HELP {name} {mhelp}")
                lines.append(f"# TYPE {name} {mtype}")
                if not series:
                    lines.append(f"{name} 0")
                else:
                    for lk, val in sorted(series.items()):
                        lines.append(f"{name}{lk} {self._fmt(val)}")

            # Histograms
            for name, series in sorted(self._histograms.items()):
                mtype, mhelp = self._meta.get(name, ("histogram", ""))
                lines.append(f"# HELP {name} {mhelp}")
                lines.append(f"# TYPE {name} {mtype}")
                for lk, entry in sorted(series.items()):
                    # Cumulative buckets
                    cumulative = 0
                    for i, bound in enumerate(self._buckets):
                        cumulative += entry["buckets"][i]
                        le = "+Inf" if math.isinf(bound) else self._fmt(bound)
                        le_label = f'le="{le}"'
                        if lk:
                            # Insert le label inside existing braces
                            combined = lk[:-1] + "," + le_label + "}"
                        else:
                            combined = "{" + le_label + "}"
                        lines.append(f"{name}_bucket{combined} {cumulative}")
                    lines.append(f"{name}_sum{lk} {self._fmt(entry['sum'])}")
                    lines.append(f"{name}_count{lk} {entry['count']}")

        lines.append("")  # trailing newline
        return "\n".join(lines)

    @staticmethod
    def _fmt(val: float) -> str:
        """Format a float for Prometheus output — drop trailing zeros."""
        if val == int(val):
            return str(int(val))
        return f"{val:.6g}"


# ═══════════════════════════════════════════════════════════════════════════
# Singleton collector instance
# ═══════════════════════════════════════════════════════════════════════════

collector = MetricsCollector()


# ═══════════════════════════════════════════════════════════════════════════
# Metrics Middleware
# ═══════════════════════════════════════════════════════════════════════════

class MetricsMiddleware(BaseHTTPMiddleware):
    """Starlette middleware that records HTTP metrics for every request.

    Tracks:
        - ``medai_http_requests_total`` (counter)
        - ``medai_http_request_duration_seconds`` (histogram)
        - ``medai_active_requests`` (gauge)
    """

    def __init__(self, app: Any, metrics: MetricsCollector | None = None):
        super().__init__(app)
        self._metrics = metrics or collector

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        method = request.method
        path = _normalise_path(request.url.path)

        # Track in-flight requests
        self._metrics.inc_gauge("medai_active_requests")

        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            # Record the error before re-raising
            self._metrics.dec_gauge("medai_active_requests")
            self._metrics.inc_counter(
                "medai_http_requests_total",
                {"method": method, "path": path, "status_code": "500"},
            )
            duration = time.perf_counter() - start
            self._metrics.observe_histogram(
                "medai_http_request_duration_seconds",
                duration,
                {"method": method, "path": path},
            )
            raise

        duration = time.perf_counter() - start
        status_code = str(response.status_code)

        self._metrics.dec_gauge("medai_active_requests")
        self._metrics.inc_counter(
            "medai_http_requests_total",
            {"method": method, "path": path, "status_code": status_code},
        )
        self._metrics.observe_histogram(
            "medai_http_request_duration_seconds",
            duration,
            {"method": method, "path": path},
        )

        return response


# ═══════════════════════════════════════════════════════════════════════════
# Metrics Router  (/metrics endpoint)
# ═══════════════════════════════════════════════════════════════════════════

metrics_router = APIRouter()


@metrics_router.get("/metrics", tags=["Metrics"])
async def prometheus_metrics(request: Request) -> Response:
    """Return all collected metrics in Prometheus text exposition format.

    The endpoint checks ``app.state.metrics`` first (the collector stored by
    the application factory) and falls back to the module-level singleton.
    """
    mc: MetricsCollector = getattr(request.app.state, "metrics", None) or collector
    body = mc.export()
    return Response(
        content=body,
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
