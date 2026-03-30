"""
MediScan AI v7.0 — Main Engine Orchestrator
Production-grade medical imaging intelligence pipeline:
  Ingestion → Preprocessing → Routing → Parallel Execution →
  Reasoning Engine → Dynamic Fusion → Self-Reflection →
  Safety Layer → Multi-Agent → Explainability → Reporting

v7.0 UPGRADES (over v6.0):
  ✅ 16 models (up from 10) — 5 new domain specialists
  ✅ Merlin (Stanford MIMI) — 3D CT specialist alongside Med3DVLM
  ✅ CheXagent-2 (Stanford AIMI) — Chest X-ray specialist (3B + 8B)
  ✅ PathGen-1.6B — Computational pathology specialist
  ✅ RETFound — Retinal foundation model (encoder-only)
  ✅ RadFM — General radiology foundation model (2D + 3D)
  ✅ Enhanced routing: domain specialists get priority for their modalities
  ✅ Modality-aware fusion weighting (CheXagent boosted for CXR, etc.)
  ✅ Multi-3D consensus: Med3DVLM + Merlin + RadFM for CT volumes
  ✅ Specialist-aware execution task builder
"""
from __future__ import annotations


import argparse
import logging
import os
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import numpy as np
import yaml
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("mediscan")

VERSION = "7.0"
_PACKAGE_DIR = Path(__file__).resolve().parent


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    raw = os.getenv(name)
    if raw is None:
        return max(minimum, default)
    try:
        return max(minimum, int(raw))
    except ValueError:
        logger.warning("Invalid integer env var %s=%r; using default %s", name, raw, default)
        return max(minimum, default)

# ── Model Capability Matrix ──────────────────────────────────────────────────
# Models that support native 3D volume input
NATIVE_3D_MODELS = {"hulu_med_7b", "hulu_med_14b", "hulu_med_32b", "med3dvlm", "merlin", "radfm"}
# Models that only support 2D images
IMAGE_ONLY_MODELS = {
    "medgemma_4b", "medgemma_27b",
    "medix_r1_8b", "medix_r1_30b",
    "biomedclip",
    "chexagent_8b", "chexagent_3b",
    "pathgen", "retfound",
}


class MediScanEngine:
    """The core MediScan AI v7.0 engine — orchestrates the complete pipeline.

    16 models across 40+ medical imaging modalities with domain-specialist routing.
    """

    def __init__(self, config_dir: Optional[str] = None):
        self.config_dir = Path(config_dir) if config_dir else _PACKAGE_DIR / "config"
        self.model_config = self._load_yaml("model_config.yaml")
        self.pipeline_config = self._load_yaml("pipeline_config.yaml")
        self.hardware_config = self._load_yaml("hardware_config.yaml")
        self._model_state_lock = threading.RLock()
        self._resident_models: OrderedDict[str, None] = OrderedDict()
        self._active_model_calls: dict[str, int] = {}
        lazy_loading = self.hardware_config.get("lazy_loading", {})
        self._max_resident_models = _env_int(
            "MEDISCAN_MAX_RESIDENT_MODELS",
            int(lazy_loading.get("max_resident_models", 0)),
            minimum=0,
        )
        self._auto_unload_after_inference = _env_bool(
            "MEDISCAN_AUTO_UNLOAD_AFTER_INFERENCE",
            bool(lazy_loading.get("auto_unload_after_inference", False)),
        )
        self._sequential_heavy_models = _env_bool(
            "MEDISCAN_SEQUENTIAL_HEAVY_MODELS",
            bool(lazy_loading.get("sequential_heavy_models", False)),
        )

        self._init_ingestion()
        self._init_preprocessing()
        self._init_models()
        self._init_routing()
        self._init_execution()
        self._init_fusion()
        self._init_governance()
        self._init_rag()
        self._init_reporting()
        self._init_translation()
        self._init_memory()
        self._init_monitoring()
        self._init_orchestration()
        self._init_intelligence()

        logger.info(f"🚀 MediScan AI v{VERSION} Engine initialized — {len(self.models)} models registered")
        logger.info(
            "MediScan residency policy: max_resident=%d auto_unload=%s sequential_heavy=%s",
            self._max_resident_models,
            self._auto_unload_after_inference,
            self._sequential_heavy_models,
        )

    def _load_yaml(self, filename: str) -> dict:
        path = self.config_dir / filename
        if path.exists():
            with open(path, "r") as f:
                return yaml.safe_load(f) or {}
        logger.warning(f"Config not found: {path}")
        return {}

    # ═══════════════════════════════════════════════════════
    #  INITIALIZATION
    # ═══════════════════════════════════════════════════════

    def _init_ingestion(self):
        from .core.ingestion.dicom_loader import DICOMLoader
        from .core.ingestion.image_loader import ImageLoader
        from .core.ingestion.video_loader import VideoLoader
        from .core.ingestion.modality_detector import ModalityDetector
        from .core.ingestion.quality_assessor import QualityAssessor, MetadataExtractor
        self.dicom_loader = DICOMLoader()
        self.image_loader = ImageLoader()
        self.video_loader = VideoLoader()
        self.modality_detector = ModalityDetector()
        self.quality_assessor = QualityAssessor()
        self.metadata_extractor = MetadataExtractor()

    def _init_preprocessing(self):
        from .core.preprocessing.monai_pipeline import MONAIPipeline
        self.preprocessor = MONAIPipeline()

    def _init_models(self):
        """v7.0: Initialize all 18 models with their wrappers."""
        from .core.models.foundation.hulu_med import HuluMedModel
        from .core.models.foundation.medgemma import MedGemmaModel
        from .core.models.foundation.medix_r1 import MediXR1Model
        from .core.models.foundation.biomedclip import BiomedCLIPModel
        from .core.models.three_d_models.med3dvlm import Med3DVLMModel
        from .core.models.three_d_models.merlin import MerlinModel
        from .core.models.specialists.chexagent import CheXagentModel
        from .core.models.specialists.pathgen import PathGenModel
        from .core.models.specialists.retfound import RETFoundModel
        from .core.models.specialists.radfm import RadFMModel
        from .core.models.reasoning.med_reasoner import MedicalReasoner

        model_configs = self.model_config.get("models", {})
        defaults = self.model_config.get("defaults", {})
        self.models = {}

        # v7.0: Complete model class map — 18 models
        model_class_map = {
            # Foundation VLMs
            "hulu_med_7b": HuluMedModel,
            "hulu_med_14b": HuluMedModel,
            "hulu_med_32b": HuluMedModel,
            "medgemma_4b": MedGemmaModel,
            "medgemma_27b": MedGemmaModel,
            # Reasoners
            "medix_r1_2b": MediXR1Model,
            "medix_r1_8b": MediXR1Model,
            "medix_r1_30b": MediXR1Model,
            # 3D Specialists
            "med3dvlm": Med3DVLMModel,
            "merlin": MerlinModel,              # v7.0: Stanford MIMI 3D CT
            # Domain Specialists
            "chexagent_8b": CheXagentModel,     # v7.0: Stanford AIMI CXR
            "chexagent_3b": CheXagentModel,     # v7.0: Lightweight CXR
            "pathgen": PathGenModel,            # v7.0: Pathology specialist
            "retfound": RETFoundModel,          # v7.0: Retinal foundation
            "radfm": RadFMModel,                # v7.0: General radiology
            # Classifiers
            "biomedclip": BiomedCLIPModel,
        }

        for key, wrapper_class in model_class_map.items():
            if key in model_configs:
                config = {**defaults, **model_configs[key]}
                model_id = config.get("model_id", key)
                self.models[key] = wrapper_class(model_id=model_id, config=config)
                logger.info(f"Model registered: {key} → {model_id}")

        self.reasoner = MedicalReasoner()
        logger.info(f"v7.0: {len(self.models)} models registered")

    def _init_routing(self):
        from .core.routing.intelligent_router import IntelligentRouter, FallbackManager
        routing_config = self.pipeline_config.get("routing", {})
        self.router = IntelligentRouter(config=routing_config)
        self.fallback_manager = FallbackManager()
        for key in self.models:
            self.router.register_available_model(key)

    def _init_execution(self):
        from .core.execution.parallel_executor import ParallelExecutor
        max_workers = self.hardware_config.get("parallel", {}).get("max_concurrent_models", 4)
        self.executor = ParallelExecutor(max_workers=max_workers)

    def _init_fusion(self):
        from .core.fusion.multi_model_fusion import (
            MultiModelFusion, ConfidenceScorer, UncertaintyEstimator,
            ContradictionDetector, AntiHallucination,
        )
        self.fusion = MultiModelFusion()
        self.confidence_scorer = ConfidenceScorer()
        self.uncertainty_estimator = UncertaintyEstimator()
        self.contradiction_detector = ContradictionDetector()
        biomedclip = self.models.get("biomedclip")
        self.anti_hallucination = AntiHallucination(biomedclip=biomedclip, rag_engine=None)

    def _init_governance(self):
        from .core.gov.governance import (
            ClinicalValidator, GuidelineChecker, RiskFlagger,
            Explainability, AuditLogger,
        )
        self.clinical_validator = ClinicalValidator()
        self.guideline_checker = GuidelineChecker()
        self.risk_flagger = RiskFlagger()
        self.explainability = Explainability()
        audit_dir = self.pipeline_config.get("governance", {}).get("audit", {}).get("log_dir", "./logs/audit")
        self.audit_logger = AuditLogger(log_dir=audit_dir)

    def _init_rag(self):
        from .core.rag.medical_rag import MedicalRAG, WebSearch
        rag_config = self.pipeline_config.get("rag", {})
        self.rag = MedicalRAG(
            collection_name=rag_config.get("collection_name", "medical_knowledge"),
            embedding_model=rag_config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
        )
        self.web_search = WebSearch()
        if hasattr(self, "anti_hallucination"):
            self.anti_hallucination.rag_engine = self.rag

    def _init_reporting(self):
        from .core.reporting.report_generator import ReportGenerator, FHIRFormatter
        self.report_generator = ReportGenerator()
        self.fhir_formatter = FHIRFormatter()

    def _init_translation(self):
        from .core.translation.sarvam_ai import SarvamTranslator, LanguageDetector
        self.translator = SarvamTranslator()
        self.language_detector = LanguageDetector()

    def _init_memory(self):
        from .core.memory.patient_history import PatientHistory, CaseTracker
        self.patient_history = PatientHistory()
        self.case_tracker = CaseTracker()

    def _init_monitoring(self):
        from .core.monitoring.monitoring import DriftDetector, OODDetector, PerformanceMetrics
        self.drift_detector = DriftDetector()
        self.ood_detector = OODDetector()
        self.performance_metrics = PerformanceMetrics()

    def _init_orchestration(self):
        from .core.orchestration.conversation_orchestrator import ConversationOrchestrator
        from .core.orchestration.response_styler import ResponseStyler
        self.orchestrator = ConversationOrchestrator(engine=self, default_mode="doctor")
        self.response_styler = ResponseStyler()

    def _init_intelligence(self):
        """v7.0: Initialize Intelligence Layer + wire LLM features."""
        from .core.intelligence.intelligence_engine import (
            MedicalReasoningEngine, DynamicFusionEngine, ClinicalSafetyLayer,
            SelfReflectionLoop, MultiAgentOrchestrator, ExplainabilityEngine,
            EnhancedMedicalRAG,
        )
        self.reasoning_engine = MedicalReasoningEngine()
        self.dynamic_fusion = DynamicFusionEngine()
        self.safety_layer = ClinicalSafetyLayer()
        self.self_reflection = SelfReflectionLoop()
        self.multi_agent = MultiAgentOrchestrator()
        self.explainability_engine = ExplainabilityEngine()
        self.enhanced_rag = EnhancedMedicalRAG()

        # v7.0 FIX: Wire LLM-powered features so they are NOT dead code
        # Use the smallest available text-capable model as the reasoning backbone
        reasoning_model = (
            self.models.get("medix_r1_2b")
            or self.models.get("medix_r1_8b")
            or self.models.get("medgemma_4b")
            or self.models.get("hulu_med_7b")
        )
        if reasoning_model:
            self.orchestrator.set_reasoning_model(reasoning_model)
            self.fusion.set_llm_judge(reasoning_model)
            logger.info(f"✅ LLM features wired → {reasoning_model.model_id}")
        else:
            logger.warning("⚠️ No text-capable model found — LLM features disabled")

        logger.info("✅ v7.0 Intelligence Layer initialized")

    # ═══════════════════════════════════════════════════════
    #  MAIN ANALYSIS PIPELINE
    # ═══════════════════════════════════════════════════════

    def analyze(
        self, file_path: str,
        question: str = "Generate a comprehensive medical report for this image.",
        target_language: str = "en", patient_id: Optional[str] = None,
        complexity: str = "standard", models_to_use: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Run the complete MediScan AI v7.0 analysis pipeline.

        14-stage pipeline:
          1. Ingestion → 2. Quality → 3. Modality Detection → 4. MONAI Preprocessing
          5. Routing → 6. Parallel Execution → 7. Reasoning Engine
          8. Dynamic Fusion → 9. Uncertainty/Contradiction → 10. Self-Reflection
          11. Clinical Safety → 12. Governance + Multi-Agent + Explainability
          13. Report Generation → 14. Translation
        """
        request_id = str(uuid4())
        pipeline_start = time.time()
        logger.info(f"═══ Analysis {request_id[:8]} started (v{VERSION}) ═══")
        logger.info(f"Input: {file_path}")

        try:
            # Step 1: Ingestion
            logger.info("Step 1/14: Ingestion")
            data = self._ingest(file_path)

            # Step 2: Modality Detection (must run BEFORE quality so
            #         modality-specific quality checks are not bypassed)
            logger.info("Step 2/14: Modality Detection")
            modality_info = self.modality_detector.detect(data)
            data["modality_info"] = modality_info
            modality = modality_info["modality"]

            # Step 3: Quality Assessment (now receives detected modality)
            logger.info("Step 3/14: Quality Assessment")
            quality = self.quality_assessor.assess(data, modality=modality)
            if not quality["is_acceptable"]:
                logger.warning(f"Quality below threshold: {quality['overall_score']:.2f}")

            # Step 4: MONAI Preprocessing
            logger.info("Step 4/14: MONAI Preprocessing")
            preprocessed = self.preprocessor.preprocess(data, modality=modality)

            # Step 5: Routing (v7.0: 18 models, domain-specialist aware)
            logger.info("Step 5/14: Intelligent Routing")
            if models_to_use:
                route = {
                    "primary": models_to_use, "secondary": [], "verifier": [],
                    "reasoner": [], "specialist_3d": [], "specialist_domain": [],
                    "generalist": [],
                }
            else:
                route = self.router.route(
                    modality=modality, file_type=data.get("type", "2d"),
                    complexity=complexity,
                )

            # Step 6: Parallel Model Execution (v7.0: 3D-aware + specialist-aware)
            logger.info("Step 6/14: Parallel Model Execution")
            tasks = self._build_execution_tasks(route, preprocessed, question, data)
            execution_results = self._execute_tasks_with_policy(tasks, timeout=300)

            # Step 7: Reasoning Engine (CoT + Knowledge Graph)
            logger.info("Step 7/14: Reasoning Engine")
            model_outputs = [
                {
                    "answer": r.get("result", {}).get("answer", ""),
                    "model": r["model_key"],
                    "confidence": r.get("result", {}).get("confidence", 0.5),
                }
                for r in execution_results if r.get("status") == "success"
            ]
            reasoning = self.reasoning_engine.reason(
                model_outputs, modality=modality, rag=self.enhanced_rag,
            )
            logger.info(
                f"  Findings: {reasoning['finding_count']} | "
                f"Contradictions: {len(reasoning['contradictions'])} | "
                f"Confidence: {reasoning['confidence']:.2f}"
            )

            # Step 8: Dynamic Fusion (v7.0: modality-aware specialist boost)
            logger.info("Step 8/14: Dynamic Fusion")
            fused = self.dynamic_fusion.fuse(
                execution_results, reasoning_output=reasoning, modality=modality,
            )
            logger.info(f"  Best: {fused['best_model']} | Confidence: {fused['confidence']:.2f}")

            # Step 9: Uncertainty & Contradiction Analysis
            logger.info("Step 9/14: Uncertainty & Contradiction Analysis")
            successful_results = [r["result"] for r in execution_results if r.get("status") == "success"]
            uncertainty = self.uncertainty_estimator.estimate(successful_results)
            fused["uncertainty_details"] = uncertainty
            contradictions = self.contradiction_detector.detect(successful_results)
            fused["contradictions"] = contradictions

            # Step 10: Self-Reflection
            logger.info("Step 10/14: Self-Reflection")
            reflection = self.self_reflection.reflect(fused.get("consensus_answer", ""))
            if reflection.get("improvements"):
                for imp in reflection["improvements"]:
                    logger.info(f"  💭 {imp}")

            # Step 11: Clinical Safety Validation
            logger.info("Step 11/14: Clinical Safety Validation")
            safety = self.safety_layer.validate(
                fused.get("consensus_answer", ""), reasoning, fused,
            )
            logger.info(
                f"  Safe: {safety['is_safe']} | Risk: {safety['risk_level']} | "
                f"Issues: {safety['issue_count']}"
            )

            # Step 12: Governance + Multi-Agent + Explainability
            logger.info("Step 12/14: Governance")
            governance = {
                "clinical_validation": self.clinical_validator.validate(fused),
                "guideline_check": self.guideline_checker.check(
                    fused.get("consensus_answer", "")
                ),
                "risk_assessment": reasoning.get(
                    "risk_assessment",
                    self.risk_flagger.flag(fused.get("consensus_answer", "")),
                ),
                "explainability": self.explainability.generate_attention_summary(
                    successful_results
                ),
                "reasoning": {
                    "chain": reasoning.get("reasoning_chain", []),
                    "differential": reasoning.get("differential_diagnosis", []),
                    "contradictions": reasoning.get("contradictions", []),
                },
                "safety": safety,
            }

            agents = self.multi_agent.orchestrate(execution_results, reasoning)
            dm = agents.get("decision_maker", {})
            if dm.get("recommendation"):
                logger.info(f"  🤖 Decision: {dm['recommendation'][:120]}")

            explanation = self.explainability_engine.explain(reasoning, fused)

            # Step 13: Report Generation
            logger.info("Step 13/14: Report Generation")
            report = self.report_generator.generate(
                fused_result=fused, modality_info=modality_info,
                governance_result=governance,
                patient_info=data.get("patient_info"),
                study_info=data.get("study_info"),
            )
            if isinstance(report, dict):
                report["reasoning"] = reasoning
                report["explanation"] = explanation
                report["reflection"] = reflection
                report["multi_agent"] = agents
                report["safety"] = safety

            # Step 14: Translation
            if target_language != "en":
                logger.info(f"Step 14/14: Translation → {target_language}")
                import asyncio
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None

                if loop and loop.is_running():
                    # Already inside an async context (e.g. FastAPI) — run in a new thread
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        report = pool.submit(
                            asyncio.run,
                            self.translator.translate_report(report, target_language)
                        ).result(timeout=60)
                else:
                    report = asyncio.run(
                        self.translator.translate_report(report, target_language)
                    )
            else:
                logger.info("Step 14/14: Translation (skipped — English)")

            pipeline_duration = time.time() - pipeline_start
            models_used = [
                r["model_key"] for r in execution_results if r.get("status") == "success"
            ]
            self.audit_logger.log_analysis(
                request_id=request_id,
                input_metadata={"file_type": data.get("type"), "modality": modality},
                models_used=models_used, results=fused,
                governance_results=governance,
            )
            if patient_id:
                self.patient_history.add_record(patient_id, report)
                self.case_tracker.create_case(report)
            self.performance_metrics.record_inference(
                pipeline_duration, "pipeline", success=True,
            )
            self.drift_detector.add_observation(fused.get("confidence", 0))
            ood = self.ood_detector.check(modality_info, fused.get("confidence", 0))
            if ood["is_ood"]:
                logger.warning(f"OOD detected: {ood['reasons']}")

            result = {
                "request_id": request_id,
                "version": VERSION,
                "report": report,
                "report_text": (
                    self.report_generator.to_text(report)
                    if isinstance(report, dict) else str(report)
                ),
                "fhir": (
                    self.fhir_formatter.format(report) if isinstance(report, dict) else None
                ),
                "governance": governance,
                "fusion": fused,
                "quality": quality,
                "modality": modality_info,
                "ood_check": ood,
                "reasoning": reasoning,
                "safety": safety,
                "explanation": explanation,
                "reflection": reflection,
                "pipeline_duration": round(pipeline_duration, 2),
                "models_used": models_used,
            }
            logger.info(
                f"═══ Analysis {request_id[:8]} completed in {pipeline_duration:.1f}s "
                f"({len(models_used)} models) ═══"
            )
            return result

        except Exception as e:
            pipeline_duration = time.time() - pipeline_start
            self.performance_metrics.record_inference(
                pipeline_duration, "pipeline", success=False,
            )
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            return {
                "request_id": request_id,
                "error": str(e),
                "pipeline_duration": round(pipeline_duration, 2),
            }

    # ═══════════════════════════════════════════════════════
    #  CONVERSATIONAL INTERFACE
    # ═══════════════════════════════════════════════════════

    def chat(
        self, user_input: str, file_path: Optional[str] = None,
        mode: Optional[str] = None, language: str = "en",
    ) -> dict[str, Any]:
        """Conversational interface via ConversationOrchestrator."""
        if mode:
            self.orchestrator.mode = mode
        return self.orchestrator.process(
            user_input=user_input, file_path=file_path,
            context={"language": language},
        )

    def analyze_conversational(
        self, file_path: str,
        question: str = "Analyze this medical image.",
        mode: str = "doctor", target_language: str = "en",
        **kwargs,
    ) -> dict[str, Any]:
        """Run analyze() + restyle via ResponseStyler."""
        result = self.analyze(
            file_path=file_path, question=question,
            target_language=target_language, **kwargs,
        )
        if "error" not in result and result.get("report"):
            result["styled_text"] = self.response_styler.rewrite(
                result["report"], mode=mode,
            )
            result["mode"] = mode
        return result

    # ═══════════════════════════════════════════════════════
    #  v7.0: PER-MODEL ROUTING + EXECUTION
    # ═══════════════════════════════════════════════════════

    def _ingest(self, file_path: str) -> dict[str, Any]:
        path = Path(file_path)
        suffix = "".join(path.suffixes).lower()
        if suffix in (".dcm",) or path.suffix.lower() == ".dcm":
            return self.dicom_loader.load(path)
        elif suffix in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
            return self.video_loader.load(path)
        else:
            return self.image_loader.load(path)

    def _build_execution_tasks(
        self, route: dict[str, list[str]],
        preprocessed: dict[str, Any], question: str,
        raw_data: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Build execution tasks — v7.0: 3D per-model + specialist routing.

        When file_type == "3d":
          HuluMed / Med3DVLM / Merlin / RadFM → native 3D (volume data)
          MedGemma / MediX-R1 / CheXagent → middle slice as 2D
          BiomedCLIP / RETFound / PathGen → first slice as 2D
        """
        tasks = []
        file_type = raw_data.get("type", "2d")
        source_path = raw_data.get("source_path", "")

        # Pre-extract 2D slices from 3D volumes
        middle_slice_pil, first_slice_pil = None, None
        if file_type == "3d":
            volume = raw_data.get("volume")
            if volume is not None:
                try:
                    mid_idx = volume.shape[0] // 2
                    mid_s = volume[mid_idx]
                    smin, smax = mid_s.min(), mid_s.max()
                    mid_norm = ((mid_s - smin) / (smax - smin + 1e-8) * 255).astype(np.uint8)
                    middle_slice_pil = Image.fromarray(mid_norm, mode="L").convert("RGB")

                    first_s = volume[0]
                    smin, smax = first_s.min(), first_s.max()
                    first_norm = ((first_s - smin) / (smax - smin + 1e-8) * 255).astype(np.uint8)
                    first_slice_pil = Image.fromarray(first_norm, mode="L").convert("RGB")
                except Exception as e:
                    logger.warning(f"Failed to extract 2D slices: {e}")

        for role, model_keys in route.items():
            for model_key in model_keys:
                if model_key not in self.models:
                    continue
                model = self.models[model_key]

                # Per-model modality adaptation
                if file_type == "3d":
                    task_images, task_modality, task_kwargs = self._adapt_3d_for_model(
                        model_key, preprocessed, raw_data, source_path,
                        middle_slice_pil, first_slice_pil,
                    )
                elif file_type == "2d":
                    pil_image = preprocessed.get("pil_image")
                    task_images = [pil_image] if pil_image else None
                    task_modality = "image"
                    task_kwargs = {"source_path": source_path}
                elif file_type == "video":
                    task_images = preprocessed.get(
                        "frames", raw_data.get("frames", [])
                    )
                    task_modality = "video"
                    task_kwargs = {
                        "source_path": source_path,
                        "video_path": source_path,
                    }
                else:
                    pil_image = preprocessed.get("pil_image")
                    task_images = [pil_image] if pil_image else None
                    task_modality = "image"
                    task_kwargs = {"source_path": source_path}

                # Build reasoning prompt for reasoner models
                task_question = question
                if role == "reasoner":
                    task_question = self.reasoner.build_reasoning_prompt(
                        question, [], scenario="diagnosis",
                    )

                def make_callable(m, imgs, txt, mod, kw, mk):
                    def fn():
                        self._prepare_model_for_inference(mk)
                        try:
                            return m.analyze(
                                images=imgs, text=txt, modality=mod, **kw,
                            )
                        except Exception as e:
                            logger.error(f"Model {mk} failed: {e}")
                            raise
                        finally:
                            self._release_model_after_inference(mk)
                    return fn

                tasks.append({
                    "model_key": model_key,
                    "callable": make_callable(
                        model, task_images, task_question,
                        task_modality, task_kwargs, model_key,
                    ),
                    "role": role,
                })

        return tasks

    def _is_heavy_model(self, model_key: str) -> bool:
        model = self.models.get(model_key)
        if model is None:
            return False

        memory_tier = str(model.config.get("memory_tier", "")).strip().lower()
        if memory_tier in {"heavy", "large", "xl", "giant"}:
            return True
        if memory_tier in {"light", "small", "medium"}:
            return False

        descriptor = f"{model_key} {model.model_id}".lower()
        return any(token in descriptor for token in ("32b", "30b", "27b", "14b"))

    def _touch_resident_model(self, model_key: str) -> None:
        self._resident_models.pop(model_key, None)
        self._resident_models[model_key] = None

    def _prepare_model_for_inference(self, model_key: str) -> None:
        model = self.models[model_key]

        with self._model_state_lock:
            self._active_model_calls[model_key] = self._active_model_calls.get(model_key, 0) + 1
            if model.is_loaded:
                self._touch_resident_model(model_key)
                return

            try:
                if self._max_resident_models > 0:
                    loaded_models = [
                        key for key in self._resident_models
                        if self.models.get(key) is not None and self.models[key].is_loaded
                    ]
                    while len(loaded_models) >= self._max_resident_models:
                        evicted_key = next(
                            (
                                key for key in loaded_models
                                if self._active_model_calls.get(key, 0) == 0
                            ),
                            None,
                        )
                        if evicted_key is None:
                            break
                        logger.info("Evicting resident MediScan model before load: %s", evicted_key)
                        self.models[evicted_key].unload()
                        self._resident_models.pop(evicted_key, None)
                        loaded_models = [
                            key for key in self._resident_models
                            if self.models.get(key) is not None and self.models[key].is_loaded
                        ]

                logger.info("Preparing MediScan model for inference: %s", model_key)
                model.load()
                self._touch_resident_model(model_key)
            except Exception:
                active = max(0, self._active_model_calls.get(model_key, 1) - 1)
                if active == 0:
                    self._active_model_calls.pop(model_key, None)
                else:
                    self._active_model_calls[model_key] = active
                raise

    def _release_model_after_inference(self, model_key: str) -> None:
        model = self.models[model_key]

        with self._model_state_lock:
            active = max(0, self._active_model_calls.get(model_key, 1) - 1)
            if active == 0:
                self._active_model_calls.pop(model_key, None)
            else:
                self._active_model_calls[model_key] = active

            should_unload = (
                self._auto_unload_after_inference
                or (self._sequential_heavy_models and self._is_heavy_model(model_key))
            )
            if should_unload and active == 0 and model.is_loaded:
                logger.info("Auto-unloading MediScan model after inference: %s", model_key)
                model.unload()
                self._resident_models.pop(model_key, None)
            elif model.is_loaded:
                self._touch_resident_model(model_key)

    def _execute_tasks_with_policy(
        self,
        tasks: list[dict[str, Any]],
        timeout: float = 300.0,
    ) -> list[dict[str, Any]]:
        if not tasks:
            return []
        if not self._sequential_heavy_models:
            return self.executor.execute_parallel(tasks, timeout=timeout)

        light_tasks = [task for task in tasks if not self._is_heavy_model(task["model_key"])]
        heavy_tasks = [task for task in tasks if self._is_heavy_model(task["model_key"])]

        results: list[dict[str, Any]] = []
        if light_tasks:
            results.extend(self.executor.execute_parallel(light_tasks, timeout=timeout))
        if heavy_tasks:
            logger.info("Executing %d heavy MediScan task(s) sequentially", len(heavy_tasks))
            results.extend(self.executor.execute_sequential(heavy_tasks, timeout=timeout))
        return results

    def _adapt_3d_for_model(
        self, model_key, preprocessed, raw_data, source_path,
        middle_slice_pil, first_slice_pil,
    ):
        """v7.0: Adapt 3D input per-model capability.

        Native 3D: hulu_med_*, med3dvlm, merlin, radfm
        2D fallback: everything else → middle or first slice
        """
        if model_key in NATIVE_3D_MODELS:
            extra_kwargs = {
                "source_path": source_path,
                "volume_array": raw_data.get("volume"),
                "nii_path": source_path,
                "nii_num_slices": 180,
                "nii_axis": 2,
            }
            return None, "3d", extra_kwargs
        elif model_key in ("biomedclip", "retfound"):
            # Classifiers/encoders: use first slice
            images = [first_slice_pil] if first_slice_pil else None
            return images, "image", {"source_path": source_path}
        else:
            # All other generative 2D models: use middle slice
            images = [middle_slice_pil] if middle_slice_pil else None
            return images, "image", {"source_path": source_path}

    # ═══════════════════════════════════════════════════════
    #  UTILITY METHODS
    # ═══════════════════════════════════════════════════════

    def load_model(self, model_key: str) -> bool:
        if model_key in self.models:
            try:
                model = self.models[model_key]
                with self._model_state_lock:
                    if not model.is_loaded:
                        if self._max_resident_models > 0:
                            loaded_models = [
                                key for key in self._resident_models
                                if self.models.get(key) is not None and self.models[key].is_loaded
                            ]
                            while len(loaded_models) >= self._max_resident_models:
                                evicted_key = next(
                                    (
                                        key for key in loaded_models
                                        if self._active_model_calls.get(key, 0) == 0
                                    ),
                                    None,
                                )
                                if evicted_key is None:
                                    break
                                self.models[evicted_key].unload()
                                self._resident_models.pop(evicted_key, None)
                                loaded_models = [
                                    key for key in self._resident_models
                                    if self.models.get(key) is not None and self.models[key].is_loaded
                                ]
                        model.load()
                    self._touch_resident_model(model_key)
                return True
            except Exception as e:
                logger.error(f"Failed to load {model_key}: {e}")
        return False

    def list_models(self) -> dict[str, Any]:
        """Return status of all registered models."""
        return {
            key: {
                "model_id": model.model_id,
                "is_loaded": model.is_loaded,
                "type": (
                    "3d_specialist" if key in NATIVE_3D_MODELS
                    else "image_only" if key in IMAGE_ONLY_MODELS
                    else "unknown"
                ),
            }
            for key, model in self.models.items()
        }

    def health_check(self) -> dict[str, Any]:
        return {
            "status": "healthy",
            "version": VERSION,
            "total_models": len(self.models),
            "models": {key: model.health_check() for key, model in self.models.items()},
            "performance": self.performance_metrics.get_metrics(),
            "drift": self.drift_detector.check_drift(),
        }

    def shutdown(self):
        logger.info(f"Shutting down MediScan AI v{VERSION}...")
        self.executor.shutdown()
        for model in self.models.values():
            if model.is_loaded:
                model.unload()
        with self._model_state_lock:
            self._resident_models.clear()
            self._active_model_calls.clear()
        logger.info("Shutdown complete.")


# ═══════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ═══════════════════════════════════════════════════════

def _safe_print(text: str) -> None:
    """Print text safely on consoles that cannot render Unicode/emoji (e.g. Windows cp1252)."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))


def main():
    # Reconfigure stdout so emoji/unicode never crashes on Windows cp1252
    if sys.platform == "win32":
        getattr(sys.stdout, "reconfigure", lambda **_: None)(errors="replace")

    parser = argparse.ArgumentParser(
        description=f"MediScan AI v{VERSION} - Medical VLM Analysis Engine (16 Models)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Models:\n"
            "  Foundation: Hulu-Med (7B/14B/32B), MedGemma (4B/27B)\n"
            "  Reasoning:  MediX-R1 (2B/8B/30B)\n"
            "  3D Volume:  Med3DVLM, Merlin (CT)\n"
            "  Specialists: CheXagent (CXR), PathGen (pathology),\n"
            "               RETFound (retinal), RadFM (radiology)\n"
            "  Classifiers: BiomedCLIP"
        ),
    )
    parser.add_argument("--file", "-f", required=True)
    parser.add_argument("--question", "-q", default="Generate a comprehensive medical report.")
    parser.add_argument("--language", "-l", default="en")
    parser.add_argument("--patient-id", "-p", default=None)
    parser.add_argument("--complexity", "-c", default="standard", choices=["simple", "standard", "complex"])
    parser.add_argument("--models", "-m", nargs="+", default=None)
    parser.add_argument("--mode", default="doctor", choices=["doctor", "patient", "research"])
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--fhir", action="store_true")
    parser.add_argument("--config-dir", default=None, help="Config directory (default: <package>/config)")
    parser.add_argument("--list-models", action="store_true", help="List all registered models and exit")
    args = parser.parse_args()

    if args.list_models:
        engine = MediScanEngine(config_dir=args.config_dir)
        for key, info in engine.list_models().items():
            status = "[OK]" if info["is_loaded"] else "[  ]"
            _safe_print(f"  {status} {key:20s} -> {info['model_id']} ({info['type']})")
        return

    if not Path(args.file).exists():
        _safe_print(f"[ERROR] File not found: {args.file}")
        sys.exit(1)

    _safe_print(f"[*] Initializing MediScan AI v{VERSION}...")
    engine = MediScanEngine(config_dir=args.config_dir)
    _safe_print(f"[*] Analyzing: {args.file}")
    result = engine.analyze_conversational(
        file_path=args.file, question=args.question, mode=args.mode,
        target_language=args.language, patient_id=args.patient_id,
        complexity=args.complexity, models_to_use=args.models,
    )

    if "error" in result:
        _safe_print(f"\n[ERROR] {result['error']}")
        sys.exit(1)
    _safe_print(f"\n{result.get('styled_text', result.get('report_text', ''))}")

    if args.output:
        import json
        output_data = result.get("fhir") if args.fhir else result.get("report", {})
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, default=str)
        _safe_print(f"\n[OK] Output saved to: {args.output}")

    engine.shutdown()


if __name__ == "__main__":
    main()
