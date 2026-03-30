"""Tests for Layer 2: Smart Router."""

from __future__ import annotations

import pytest

from core.routing.smart_router import SmartRouter, FallbackManager


class TestSmartRouter:
    """Tests for SmartRouter."""

    def setup_method(self):
        self.router = SmartRouter()
        # Register some models
        for key in ["deepseek_r1", "openbiollm_70b", "biomistral_7b",
                     "meditron_70b", "clinical_camel_70b", "med42_70b",
                     "mellama_13b", "chatdoctor"]:
            self.router.register_model(key)

    def test_simple_qa_routing(self):
        route = self.router.route("simple_qa", complexity="simple")
        assert "biomistral_7b" in route.get("primary", [])
        assert route["enable_rag"] is False  # Simple = no RAG

    def test_diagnosis_routing(self):
        route = self.router.route("diagnosis", complexity="standard")
        assert len(route.get("primary", [])) > 0
        assert route["enable_rag"] is True

    def test_emergency_routing(self):
        route = self.router.route("emergency", complexity="complex")
        primary = route.get("primary", [])
        verifier = route.get("verifier", [])
        assert len(primary) > 0
        assert len(verifier) > 0  # Emergency gets verifiers

    def test_forced_models(self):
        route = self.router.route(
            "diagnosis",
            force_models=["biomistral_7b"],
        )
        assert route["primary"] == ["biomistral_7b"]
        assert route["source"] == "forced"

    def test_complexity_simple_no_verifier(self):
        route = self.router.route("diagnosis", complexity="simple")
        assert route.get("verifier", []) == []

    def test_unavailable_model_filtered(self):
        router = SmartRouter()
        router.register_model("biomistral_7b")
        route = router.route("diagnosis")
        # deepseek_r1 not registered, should be filtered out
        all_models = (
            route.get("primary", []) +
            route.get("medical", []) +
            route.get("verifier", []) +
            route.get("reasoning", [])
        )
        assert "deepseek_r1" not in all_models


class TestFallbackManager:
    """Tests for FallbackManager."""

    def setup_method(self):
        self.fallback = FallbackManager()

    def test_get_fallback(self):
        fb = self.fallback.get_fallback("deepseek_r1")
        assert fb is not None
        assert fb in ["openbiollm_70b", "meditron_70b", "biomistral_7b"]

    def test_no_fallback_for_last_resort(self):
        fb = self.fallback.get_fallback("biomistral_7b")
        assert fb is None

    def test_auto_disable_after_failures(self):
        for _ in range(5):
            self.fallback.record_failure("openbiollm_70b")
        assert "openbiollm_70b" in self.fallback.disabled_models
        # Fallback should skip disabled models
        fb = self.fallback.get_fallback("deepseek_r1")
        assert fb != "openbiollm_70b"

    def test_success_resets_failure_count(self):
        self.fallback.record_failure("openbiollm_70b")
        self.fallback.record_failure("openbiollm_70b")
        assert self.fallback.failure_counts["openbiollm_70b"] == 2
        self.fallback.record_success("openbiollm_70b")
        assert self.fallback.failure_counts["openbiollm_70b"] == 0

    def test_reset(self):
        self.fallback.record_failure("openbiollm_70b")
        self.fallback.reset()
        assert len(self.fallback.failure_counts) == 0
        assert len(self.fallback.disabled_models) == 0
