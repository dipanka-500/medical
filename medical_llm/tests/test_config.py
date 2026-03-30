"""Tests for configuration loading and key alignment."""

from __future__ import annotations

import yaml
import pytest
from pathlib import Path


CONFIG_DIR = Path(__file__).parent.parent / "config"


class TestModelConfig:
    """Verify model_config.yaml is well-formed."""

    def setup_method(self):
        with open(CONFIG_DIR / "model_config.yaml", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

    def test_defaults_present(self):
        assert "defaults" in self.config
        defaults = self.config["defaults"]
        assert "trust_remote_code" in defaults
        assert defaults["trust_remote_code"] is False  # Security default

    def test_all_models_have_required_fields(self):
        models = self.config.get("models", {})
        for key, model in models.items():
            assert "model_id" in model, f"{key} missing model_id"
            assert "role" in model, f"{key} missing role"
            assert "weight" in model, f"{key} missing weight"

    def test_deepseek_uses_distilled(self):
        ds = self.config["models"]["deepseek_r1"]
        assert "Distill" in ds["model_id"], (
            "DeepSeek should use distilled variant for Transformers compat"
        )

    def test_mellama_correct_org(self):
        ml = self.config["models"]["mellama_13b"]
        assert "clinicalnlplab" in ml["model_id"], (
            "Me-LLaMA lives under clinicalnlplab, not meta-llama"
        )

    def test_embedding_config(self):
        assert "embedding" in self.config
        assert "model_id" in self.config["embedding"]

    def test_priority_order(self):
        assert "priority_order" in self.config
        assert len(self.config["priority_order"]) > 0


class TestPipelineConfig:
    """Verify pipeline_config.yaml is well-formed and keys match code."""

    def setup_method(self):
        with open(CONFIG_DIR / "pipeline_config.yaml", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

    def test_execution_keys(self):
        exec_cfg = self.config.get("execution", {})
        assert "max_concurrent_models" in exec_cfg
        assert "model_timeout_seconds" in exec_cfg
        assert "retry_on_failure" in exec_cfg

    def test_governance_audit_keys(self):
        gov = self.config.get("governance", {})
        audit = gov.get("audit", {})
        assert "log_dir" in audit
        assert "enabled" in audit

    def test_rag_toggle_keys(self):
        rag = self.config.get("rag", {})
        assert "enable_web_search" in rag
        assert "enable_pubmed" in rag
        assert "search_architecture" in rag
        assert "vector_store" in rag

    def test_safety_config(self):
        safety = self.config.get("safety", {})
        assert "enable_hallucination_check" in safety
        assert "risk_levels" in safety

    def test_routing_table(self):
        routing = self.config.get("routing", {})
        table = routing.get("routing_table", {})
        assert "diagnosis" in table
        assert "emergency" in table
        # Each route should have primary key
        for route_name, route_def in table.items():
            assert "primary" in route_def, f"Route {route_name} missing 'primary'"
