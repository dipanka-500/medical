"""Smoke tests — verify imports and basic construction without GPU/models."""

from __future__ import annotations

import pytest
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestImports:
    """Verify all modules import cleanly."""

    def test_import_main(self):
        import main
        assert hasattr(main, "MedicalLLMEngine")
        assert hasattr(main, "MODEL_REGISTRY")

    def test_import_app(self):
        pytest.importorskip("fastapi")
        import app
        assert hasattr(app, "app")
        assert hasattr(app, "create_app")

    def test_import_input_layer(self):
        from core.input.query_classifier import QueryClassifier
        from core.input.symptom_extractor import SymptomExtractor
        from core.input.negation_detector import NegationDetector
        from core.input.medical_ner import MedicalNER, MedicalEntity

    def test_import_routing(self):
        from core.routing.smart_router import SmartRouter, FallbackManager

    def test_import_models(self):
        from core.models.base_model import BaseLLM, HuggingFaceLLM, VLLMEngine
        from core.models.reasoning.deepseek_engine import DeepSeekEngine
        from core.models.medical.meditron_engine import MeditronEngine
        from core.models.medical.mellama_engine import MeLLaMAEngine
        from core.models.medical.openbiollm_engine import OpenBioLLMEngine
        from core.models.medical.pmc_llama_engine import PMCLLaMAEngine
        from core.models.clinical.biomistral_engine import BioMistralEngine
        from core.models.clinical.clinical_camel_engine import ClinicalCamelEngine
        from core.models.clinical.med42_engine import Med42Engine
        from core.models.conversational.chatdoctor_engine import ChatDoctorEngine

    def test_import_rag(self):
        from core.rag.medical_rag import MedicalRAG
        from core.rag.knowledge_base import KnowledgeBase
        from core.rag.pubmed_fetcher import PubMedFetcher
        from core.rag.retrieval_pipeline import MedicalRetrievalPipeline
        from core.rag.web_search import WebSearch

    def test_import_fusion(self):
        from core.fusion.meta_fusion import (
            MetaFusion,
            UncertaintyEstimator,
            ContradictionDetector,
        )

    def test_import_safety(self):
        from core.safety.safety import (
            HallucinationDetector,
            DrugInteractionChecker,
            ClinicalValidator,
            RiskFlagger,
        )

    def test_import_response(self):
        from core.response.report_generator import ReportGenerator, ResponseStyler

    def test_import_execution(self):
        from core.execution.parallel_executor import ParallelExecutor

    def test_import_governance(self):
        from core.governance.audit import AuditLogger


class TestConstruction:
    """Verify components can be constructed without GPU/models."""

    def test_construct_classifier(self):
        from core.input.query_classifier import QueryClassifier
        c = QueryClassifier()
        assert c.classify("hello")["category"] == "conversational"

    def test_construct_symptom_extractor(self):
        from core.input.symptom_extractor import SymptomExtractor
        s = SymptomExtractor()
        result = s.extract_all("chest pain")
        assert result["symptom_count"] >= 1

    def test_construct_negation_detector(self):
        from core.input.negation_detector import NegationDetector
        n = NegationDetector()
        result = n.detect("no pneumonia", "pneumonia")
        assert result.is_negated is True

    def test_construct_router(self):
        from core.routing.smart_router import SmartRouter
        r = SmartRouter()
        r.register_model("biomistral_7b")
        route = r.route("simple_qa")
        assert "primary" in route

    def test_construct_fusion(self):
        from core.fusion.meta_fusion import MetaFusion
        f = MetaFusion()
        result = f.fuse([{"text": "test", "model": "m", "role": "primary", "weight": 0.5}])
        assert result["text"] == "test"

    def test_construct_drug_checker(self):
        from core.safety.safety import DrugInteractionChecker
        d = DrugInteractionChecker()
        result = d.check_interactions(["warfarin", "aspirin"])
        assert result["has_interactions"] is True

    def test_construct_risk_flagger(self):
        from core.safety.safety import RiskFlagger
        r = RiskFlagger()
        result = r.flag("No evidence of stroke. Stable condition.")
        assert result["risk_level"] in ("routine", "urgent", "emergent")

    def test_construct_executor(self):
        from core.execution.parallel_executor import ParallelExecutor
        e = ParallelExecutor(max_workers=2, timeout=30)
        results = e.execute([
            {"name": "test", "fn": lambda: {"text": "ok"}, "kwargs": {}},
        ])
        assert len(results) == 1

    def test_construct_report_generator(self):
        from core.response.report_generator import ReportGenerator
        r = ReportGenerator()
        report = r.generate(
            fused_result={"text": "Test diagnosis", "confidence": 0.8, "agreement_score": 0.7},
            safety_result={"risk_level": "routine", "is_valid": True},
        )
        assert "report_text" in report

    def test_construct_retrieval_pipeline(self):
        from core.rag.retrieval_pipeline import MedicalRetrievalPipeline

        pipeline = MedicalRetrievalPipeline(config={"cache": {"ttl_seconds": 60}})
        assert pipeline.query_analyzer is not None

    def test_model_registry_complete(self):
        from main import MODEL_REGISTRY
        expected = [
            "deepseek_r1", "meditron_70b", "mellama_13b", "pmc_llama_13b",
            "openbiollm_70b", "biomistral_7b", "clinical_camel_70b",
            "med42_70b", "chatdoctor",
        ]
        for key in expected:
            assert key in MODEL_REGISTRY, f"Missing model registry entry: {key}"
