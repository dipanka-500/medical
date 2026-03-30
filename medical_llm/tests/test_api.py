"""Tests for the production FastAPI surface."""

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

import app as app_module


def _build_test_app(monkeypatch, settings: app_module.APISettings):
    calls: dict[str, object] = {}

    def fake_initialize(self, models_to_load=None):
        self._is_initialized = True
        calls["initialized"] = calls.get("initialized", 0) + 1

    def fake_ingest(self):
        calls["ingested"] = True
        return 0

    def fake_analyze(
        self,
        query,
        mode="doctor",
        enable_rag=True,
        force_models=None,
        use_cache=True,
        session_id=None,
    ):
        calls["analyze"] = {
            "query": query,
            "mode": mode,
            "enable_rag": enable_rag,
            "force_models": force_models,
            "use_cache": use_cache,
            "session_id": session_id,
        }
        return {
            "report_text": "ok",
            "confidence": 0.9,
            "agreement": 0.8,
            "risk_level": "routine",
            "audit_id": "test_audit",
            "execution_time": 0.01,
            "input_analysis": {},
            "routing": {},
            "model_results": [],
            "fusion": {},
            "safety": {},
            "rag_evidence": [],
            "sources": [],
            "retrieval": {"query_analysis": {}, "sources": []},
        }

    monkeypatch.setattr(app_module.MedicalLLMEngine, "initialize", fake_initialize)
    monkeypatch.setattr(app_module.MedicalLLMEngine, "ingest_built_in_knowledge", fake_ingest)
    monkeypatch.setattr(app_module.MedicalLLMEngine, "analyze", fake_analyze)
    monkeypatch.setattr(
        app_module.MedicalLLMEngine,
        "get_model_status",
        lambda self: {"deepseek_r1": {"is_loaded": False}},
    )
    monkeypatch.setattr(
        app_module.MedicalLLMEngine,
        "get_conversation_session_count",
        lambda self: 0,
    )
    monkeypatch.setattr(
        app_module.MedicalLLMEngine,
        "clear_conversation",
        lambda self, session_id: session_id == "session-1",
    )
    monkeypatch.setattr(
        app_module.MedicalLLMEngine,
        "get_state_backend_status",
        lambda self: {
            "mode": "memory",
            "redis_configured": False,
            "shared_state_enabled": False,
            "cache_ttl_seconds": 300,
            "session_ttl_seconds": 3600,
        },
    )

    return app_module.create_app(settings), calls


class TestApi:
    """API behavior tests with a mocked engine."""

    def test_health_and_ready(self, monkeypatch):
        settings = app_module.APISettings(init_on_startup=False)
        test_app, _ = _build_test_app(monkeypatch, settings)

        with TestClient(test_app) as client:
            health = client.get("/health")
            ready = client.get("/ready")

        assert health.status_code == 200
        assert health.json()["status"] == "ok"
        assert ready.status_code == 503
        assert ready.json()["ready"] is False

    def test_analyze_requires_api_key(self, monkeypatch):
        settings = app_module.APISettings(
            api_key="secret",
            init_on_startup=False,
        )
        test_app, _ = _build_test_app(monkeypatch, settings)

        with TestClient(test_app) as client:
            response = client.post("/analyze", json={"query": "hello"})

        assert response.status_code == 401

    def test_analyze_lazy_initializes_and_passes_session(self, monkeypatch):
        settings = app_module.APISettings(
            api_key="secret",
            init_on_startup=False,
            ingest_builtin_knowledge=True,
        )
        test_app, calls = _build_test_app(monkeypatch, settings)

        with TestClient(test_app) as client:
            response = client.post(
                "/analyze",
                headers={"X-API-Key": "secret"},
                json={
                    "query": "What are the side effects of metformin?",
                    "mode": "patient",
                    "session_id": "session-1",
                    "use_cache": False,
                },
            )
            ready = client.get("/ready")

        assert response.status_code == 200
        assert calls["initialized"] == 1
        assert calls.get("ingested") is True
        assert calls["analyze"]["session_id"] == "session-1"
        assert calls["analyze"]["mode"] == "patient"
        assert ready.status_code == 200
        assert ready.json()["ready"] is True
        assert ready.json()["state_backend"]["mode"] == "memory"

    def test_clear_session(self, monkeypatch):
        settings = app_module.APISettings(api_key="secret", init_on_startup=False)
        test_app, _ = _build_test_app(monkeypatch, settings)

        with TestClient(test_app) as client:
            response = client.delete(
                "/sessions/session-1",
                headers={"X-API-Key": "secret"},
            )

        assert response.status_code == 200
        assert response.json() == {"session_id": "session-1", "cleared": True}

    def test_status_includes_limiter_and_state_backend(self, monkeypatch):
        settings = app_module.APISettings(api_key="secret", init_on_startup=False)
        test_app, _ = _build_test_app(monkeypatch, settings)

        async def fake_stats():
            return {"active": 0, "waiting": 0, "distributed_enabled": False}

        test_app.state.request_limiter.get_stats = fake_stats

        with TestClient(test_app) as client:
            response = client.get("/status", headers={"X-API-Key": "secret"})

        assert response.status_code == 200
        payload = response.json()
        assert payload["state_backend"]["mode"] == "memory"
        assert payload["request_limiter"]["distributed_enabled"] is False

    def test_analyze_sheds_load_when_queue_is_full(self, monkeypatch):
        settings = app_module.APISettings(api_key="secret", init_on_startup=False)
        test_app, _ = _build_test_app(monkeypatch, settings)

        @asynccontextmanager
        async def overloaded_slot():
            raise OverflowError("AI request queue is full.")
            yield

        test_app.state.request_limiter.slot = overloaded_slot

        with TestClient(test_app) as client:
            response = client.post(
                "/analyze",
                headers={"X-API-Key": "secret"},
                json={"query": "hello"},
            )

        assert response.status_code == 503
        assert response.json()["detail"] == "AI request queue is full."
