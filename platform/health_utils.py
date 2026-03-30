from __future__ import annotations

from typing import Any


def engine_health_endpoint(engine_name: str, base_url: str) -> str:
    """Choose the most meaningful health endpoint for an engine."""
    if engine_name == "medical_llm":
        return f"{base_url}/ready"
    return f"{base_url}/health"


def normalize_engine_health(
    engine_name: str,
    status_code: int,
    payload: dict[str, Any] | None,
) -> tuple[str, str | None]:
    """Normalize heterogeneous engine health payloads into a common status."""
    raw_status = ""
    if isinstance(payload, dict):
        raw_status = str(payload.get("status", "")).strip().lower()

    normalized = "healthy" if 200 <= status_code < 300 else "unhealthy"

    if engine_name == "medical_llm":
        ready = payload.get("ready") if isinstance(payload, dict) else None
        initialized = payload.get("engine_initialized") if isinstance(payload, dict) else None

        if ready is True:
            normalized = "healthy"
        elif ready is False:
            normalized = "unhealthy" if status_code >= 500 else "degraded"
        elif initialized is False:
            normalized = "degraded"
        elif raw_status in {"ready", "healthy", "ok"}:
            normalized = "healthy"
        elif raw_status in {"initializing", "degraded"}:
            normalized = "degraded"
        elif raw_status.startswith("error") or raw_status == "unhealthy":
            normalized = "unhealthy"
        return normalized, raw_status or None

    if raw_status in {"ok", "healthy", "ready", "alive"}:
        normalized = "healthy" if 200 <= status_code < 300 else "unhealthy"
    elif raw_status in {"degraded", "initializing"}:
        normalized = "degraded"
    elif raw_status.startswith("error") or raw_status == "unhealthy":
        normalized = "unhealthy"

    return normalized, raw_status or None
