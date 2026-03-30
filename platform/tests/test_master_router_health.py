from __future__ import annotations

from health_utils import engine_health_endpoint, normalize_engine_health


def test_medical_llm_health_uses_ready_endpoint() -> None:
    endpoint = engine_health_endpoint("medical_llm", "http://medical-llm:8002")
    assert endpoint == "http://medical-llm:8002/ready"


def test_medical_llm_not_ready_is_unhealthy() -> None:
    status, raw = normalize_engine_health(
        "medical_llm",
        503,
        {"status": "initializing", "ready": False},
    )
    assert status == "unhealthy"
    assert raw == "initializing"


def test_medical_llm_uninitialized_health_is_degraded() -> None:
    status, raw = normalize_engine_health(
        "medical_llm",
        200,
        {"status": "ok", "engine_initialized": False},
    )
    assert status == "degraded"
    assert raw == "ok"


def test_ocr_degraded_payload_is_not_marked_healthy() -> None:
    status, raw = normalize_engine_health(
        "mediscan_ocr",
        200,
        {"status": "degraded", "ready_backends": []},
    )
    assert status == "degraded"
    assert raw == "degraded"


def test_engine_error_payload_is_unhealthy_even_with_http_200() -> None:
    status, raw = normalize_engine_health(
        "mediscan_vlm",
        200,
        {"status": "error: model load failed"},
    )
    assert status == "unhealthy"
    assert raw == "error: model load failed"
