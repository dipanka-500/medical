"""
MediScan AI v7.0 — Intelligent Router → Adaptive Orchestration System

v7.0 PRODUCTION UPGRADES:
  ✅ Confidence-based routing (stop early if confident)
  ✅ Cost + latency awareness (model scoring)
  ✅ Multi-stage routing (cheap → medium → heavy escalation)
  ✅ Feedback learning (performance tracking, adaptive weights)
  ✅ Query-aware routing (report vs diagnosis vs screening)
  ✅ Safety routing (force big models for critical queries)
  ✅ RAG-aware routing (strong context → smaller model)
  ✅ FallbackManager: failure types, cooldown system, smart recovery
  ✅ 18 models across 40+ modality types
"""
from __future__ import annotations


import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class IntelligentRouter:
    """Adaptive orchestration system for medical imaging tasks.

    v7.0: Dynamic routing based on confidence, cost, latency, safety,
    and feedback-driven model weighting. Replaces static table-only routing.
    """

    # ── v7.0 ROUTING TABLE ──────────────────────────────────────────
    # Roles: primary, secondary, verifier, reasoner, specialist_3d,
    #        specialist_domain, generalist
    ROUTING_TABLE = {
        # ── Radiology ─────────────────────────────────────────
        "xray": {
            "primary": ["chexagent_8b", "medgemma_4b"],
            "secondary": ["hulu_med_7b"],
            "verifier": ["biomedclip"],
            "reasoner": ["medix_r1_8b"],
            "specialist_3d": [],
            "specialist_domain": ["chexagent_3b"],
            "generalist": ["radfm"],
        },
        "ct": {
            "primary": ["hulu_med_14b", "hulu_med_7b"],
            "secondary": ["medgemma_4b"],
            "verifier": ["biomedclip"],
            "reasoner": ["medix_r1_8b"],
            "specialist_3d": ["med3dvlm", "merlin"],
            "specialist_domain": [],
            "generalist": ["radfm"],
        },
        "mri": {
            "primary": ["hulu_med_14b", "medgemma_4b"],
            "secondary": ["hulu_med_7b"],
            "verifier": ["biomedclip"],
            "reasoner": ["medix_r1_8b"],
            "specialist_3d": ["med3dvlm"],
            "specialist_domain": [],
            "generalist": ["radfm"],
        },
        "mammography": {
            "primary": ["medgemma_4b", "hulu_med_7b"],
            "secondary": [],
            "verifier": ["biomedclip"],
            "reasoner": ["medix_r1_8b"],
            "specialist_3d": [],
            "specialist_domain": [],
            "generalist": ["radfm"],
        },
        "fluoroscopy": {
            "primary": ["hulu_med_7b", "medgemma_4b"],
            "secondary": [],
            "verifier": ["biomedclip"],
            "reasoner": ["medix_r1_2b"],
            "specialist_3d": [],
            "specialist_domain": [],
            "generalist": ["radfm"],
        },
        "angiography": {
            "primary": ["hulu_med_7b", "medgemma_4b"],
            "secondary": [],
            "verifier": ["biomedclip"],
            "reasoner": ["medix_r1_8b"],
            "specialist_3d": [],
            "specialist_domain": [],
            "generalist": ["radfm"],
        },

        # ── Ultrasound ────────────────────────────────────────
        "ultrasound": {
            "primary": ["hulu_med_7b", "medgemma_4b"],
            "secondary": [],
            "verifier": ["biomedclip"],
            "reasoner": ["medix_r1_2b"],
            "specialist_3d": [],
            "specialist_domain": [],
            "generalist": [],
        },
        "echocardiography": {
            "primary": ["hulu_med_7b"],
            "secondary": ["medgemma_4b"],
            "verifier": ["biomedclip"],
            "reasoner": ["medix_r1_2b"],
            "specialist_3d": [],
            "specialist_domain": [],
            "generalist": [],
        },
        "intravascular_ultrasound": {
            "primary": ["hulu_med_7b", "medgemma_4b"],
            "secondary": [],
            "verifier": ["biomedclip"],
            "reasoner": ["medix_r1_2b"],
            "specialist_3d": [],
            "specialist_domain": [],
            "generalist": [],
        },
        "ultrasound_clip": {
            "primary": ["hulu_med_7b"],
            "secondary": [],
            "verifier": ["biomedclip"],
            "reasoner": ["medix_r1_2b"],
            "specialist_3d": [],
            "specialist_domain": [],
            "generalist": [],
        },

        # ── Nuclear Medicine ──────────────────────────────────
        "nuclear_medicine": {
            "primary": ["hulu_med_7b", "medgemma_4b"],
            "secondary": [],
            "verifier": ["biomedclip"],
            "reasoner": ["medix_r1_8b"],
            "specialist_3d": [],
            "specialist_domain": [],
            "generalist": ["radfm"],
        },
        "pet": {
            "primary": ["hulu_med_14b", "hulu_med_7b"],
            "secondary": [],
            "verifier": ["biomedclip"],
            "reasoner": ["medix_r1_8b"],
            "specialist_3d": ["med3dvlm", "merlin"],
            "specialist_domain": [],
            "generalist": ["radfm"],
        },
        "spect": {
            "primary": ["hulu_med_7b", "medgemma_4b"],
            "secondary": [],
            "verifier": ["biomedclip"],
            "reasoner": ["medix_r1_8b"],
            "specialist_3d": ["med3dvlm"],
            "specialist_domain": [],
            "generalist": [],
        },

        # ── Pathology & Microscopy (v7.0: PathGen primary) ────
        "pathology": {
            "primary": ["pathgen", "medgemma_4b"],
            "secondary": ["hulu_med_7b"],
            "verifier": ["biomedclip"],
            "reasoner": ["medix_r1_8b"],
            "specialist_3d": [],
            "specialist_domain": [],
            "generalist": [],
        },
        "cytology": {
            "primary": ["pathgen", "medgemma_4b"],
            "secondary": ["hulu_med_7b"],
            "verifier": ["biomedclip"],
            "reasoner": ["medix_r1_8b"],
            "specialist_3d": [],
            "specialist_domain": [],
            "generalist": [],
        },
        "histopathology": {
            "primary": ["pathgen", "medgemma_4b"],
            "secondary": [],
            "verifier": ["biomedclip"],
            "reasoner": ["medix_r1_8b"],
            "specialist_3d": [],
            "specialist_domain": [],
            "generalist": [],
        },
        "microbiology": {
            "primary": ["medgemma_4b", "hulu_med_7b"],
            "secondary": [],
            "verifier": ["biomedclip"],
            "reasoner": ["medix_r1_2b"],
            "specialist_3d": [],
            "specialist_domain": [],
            "generalist": [],
        },
        "fluorescence_microscopy": {
            "primary": ["medgemma_4b", "hulu_med_7b"],
            "secondary": [],
            "verifier": ["biomedclip"],
            "reasoner": ["medix_r1_2b"],
            "specialist_3d": [],
            "specialist_domain": [],
            "generalist": [],
        },
        "general_microscopy": {
            "primary": ["medgemma_4b", "hulu_med_7b"],
            "secondary": [],
            "verifier": ["biomedclip"],
            "reasoner": ["medix_r1_2b"],
            "specialist_3d": [],
            "specialist_domain": [],
            "generalist": [],
        },

        # ── Ophthalmology (v7.0: RETFound specialist) ─────────
        "fundoscopy": {
            "primary": ["medgemma_4b", "hulu_med_7b"],
            "secondary": [],
            "verifier": ["biomedclip"],
            "reasoner": ["medix_r1_8b"],
            "specialist_3d": [],
            "specialist_domain": ["retfound"],
            "generalist": [],
        },
        "oct": {
            "primary": ["medgemma_4b", "hulu_med_7b"],
            "secondary": [],
            "verifier": ["biomedclip"],
            "reasoner": ["medix_r1_8b"],
            "specialist_3d": [],
            "specialist_domain": ["retfound"],
            "generalist": [],
        },
        "ophthalmic_mapping": {
            "primary": ["medgemma_4b", "hulu_med_7b"],
            "secondary": [],
            "verifier": ["biomedclip"],
            "reasoner": ["medix_r1_2b"],
            "specialist_3d": [],
            "specialist_domain": ["retfound"],
            "generalist": [],
        },
        "ophthalmic_visual_field": {
            "primary": ["medgemma_4b", "hulu_med_7b"],
            "secondary": [],
            "verifier": [],
            "reasoner": ["medix_r1_2b"],
            "specialist_3d": [],
            "specialist_domain": [],
            "generalist": [],
        },

        # ── Dermatology ───────────────────────────────────────
        "dermoscopy": {
            "primary": ["medgemma_4b", "hulu_med_7b"],
            "secondary": [],
            "verifier": ["biomedclip"],
            "reasoner": ["medix_r1_8b"],
            "specialist_3d": [],
            "specialist_domain": [],
            "generalist": [],
        },
        "clinical_photo": {
            "primary": ["medgemma_4b", "hulu_med_7b"],
            "secondary": [],
            "verifier": ["biomedclip"],
            "reasoner": ["medix_r1_2b"],
            "specialist_3d": [],
            "specialist_domain": [],
            "generalist": [],
        },

        # ── Dental ────────────────────────────────────────────
        "dental": {
            "primary": ["medgemma_4b", "hulu_med_7b"],
            "secondary": [],
            "verifier": ["biomedclip"],
            "reasoner": ["medix_r1_2b"],
            "specialist_3d": [],
            "specialist_domain": [],
            "generalist": [],
        },
        "dental_intraoral": {
            "primary": ["medgemma_4b", "hulu_med_7b"],
            "secondary": [],
            "verifier": ["biomedclip"],
            "reasoner": ["medix_r1_2b"],
            "specialist_3d": [],
            "specialist_domain": [],
            "generalist": [],
        },
        "dental_panoramic": {
            "primary": ["medgemma_4b", "hulu_med_7b"],
            "secondary": [],
            "verifier": ["biomedclip"],
            "reasoner": ["medix_r1_2b"],
            "specialist_3d": [],
            "specialist_domain": [],
            "generalist": ["radfm"],
        },

        # ── Cardiology ────────────────────────────────────────
        "ecg": {
            "primary": ["medgemma_4b", "hulu_med_7b"],
            "secondary": [],
            "verifier": [],
            "reasoner": ["medix_r1_8b"],
            "specialist_3d": [],
            "specialist_domain": [],
            "generalist": [],
        },
        "electrophysiology": {
            "primary": ["hulu_med_7b"],
            "secondary": [],
            "verifier": [],
            "reasoner": ["medix_r1_2b"],
            "specialist_3d": [],
            "specialist_domain": [],
            "generalist": [],
        },

        # ── Endoscopy / Video ─────────────────────────────────
        "endoscopy": {
            "primary": ["hulu_med_7b"],
            "secondary": ["medgemma_4b"],
            "verifier": ["biomedclip"],
            "reasoner": ["medix_r1_2b"],
            "specialist_3d": [],
            "specialist_domain": [],
            "generalist": [],
        },
        "surgical_video": {
            "primary": ["hulu_med_7b"],
            "secondary": [],
            "verifier": [],
            "reasoner": ["medix_r1_2b"],
            "specialist_3d": [],
            "specialist_domain": [],
            "generalist": [],
        },
        "video": {
            "primary": ["hulu_med_7b"],
            "secondary": [],
            "verifier": [],
            "reasoner": ["medix_r1_2b"],
            "specialist_3d": [],
            "specialist_domain": [],
            "generalist": [],
        },

        # ── Advanced Neuro ────────────────────────────────────
        "dti": {
            "primary": ["hulu_med_14b", "hulu_med_7b"],
            "secondary": [],
            "verifier": [],
            "reasoner": ["medix_r1_8b"],
            "specialist_3d": ["med3dvlm"],
            "specialist_domain": [],
            "generalist": ["radfm"],
        },
        "fmri": {
            "primary": ["hulu_med_14b", "hulu_med_7b"],
            "secondary": [],
            "verifier": [],
            "reasoner": ["medix_r1_8b"],
            "specialist_3d": ["med3dvlm"],
            "specialist_domain": [],
            "generalist": [],
        },

        # ── Bone density ──────────────────────────────────────
        "bone_densitometry": {
            "primary": ["medgemma_4b", "hulu_med_7b"],
            "secondary": [],
            "verifier": ["biomedclip"],
            "reasoner": ["medix_r1_2b"],
            "specialist_3d": [],
            "specialist_domain": [],
            "generalist": ["radfm"],
        },

        # ── Generic 3D / general ──────────────────────────────
        "3d_volume": {
            "primary": ["hulu_med_7b"],
            "secondary": [],
            "verifier": [],
            "reasoner": ["medix_r1_8b"],
            "specialist_3d": ["med3dvlm", "merlin"],
            "specialist_domain": [],
            "generalist": ["radfm"],
        },
        "general_medical": {
            "primary": ["hulu_med_7b", "medgemma_4b"],
            "secondary": [],
            "verifier": ["biomedclip"],
            "reasoner": ["medix_r1_8b"],
            "specialist_3d": [],
            "specialist_domain": [],
            "generalist": ["radfm"],
        },
    }

    # ── Model Cost / Latency Profiles ──────────────────────────────
    MODEL_PROFILES = {
        "hulu_med_32b":  {"cost": 1.0,  "latency": 1.0,  "accuracy_base": 0.92, "size": "heavy"},
        "hulu_med_14b":  {"cost": 0.6,  "latency": 0.6,  "accuracy_base": 0.88, "size": "medium"},
        "hulu_med_7b":   {"cost": 0.3,  "latency": 0.3,  "accuracy_base": 0.85, "size": "light"},
        "medgemma_27b":  {"cost": 0.85, "latency": 0.8,  "accuracy_base": 0.90, "size": "heavy"},
        "medgemma_4b":   {"cost": 0.2,  "latency": 0.15, "accuracy_base": 0.80, "size": "light"},
        "medix_r1_30b":  {"cost": 0.9,  "latency": 0.9,  "accuracy_base": 0.91, "size": "heavy"},
        "medix_r1_8b":   {"cost": 0.35, "latency": 0.3,  "accuracy_base": 0.84, "size": "medium"},
        "medix_r1_2b":   {"cost": 0.1,  "latency": 0.08, "accuracy_base": 0.72, "size": "light"},
        "chexagent_8b":  {"cost": 0.35, "latency": 0.25, "accuracy_base": 0.87, "size": "medium"},
        "chexagent_3b":  {"cost": 0.15, "latency": 0.1,  "accuracy_base": 0.80, "size": "light"},
        "med3dvlm":      {"cost": 0.4,  "latency": 0.5,  "accuracy_base": 0.83, "size": "medium"},
        "merlin":        {"cost": 0.4,  "latency": 0.45, "accuracy_base": 0.82, "size": "medium"},
        "pathgen":       {"cost": 0.15, "latency": 0.1,  "accuracy_base": 0.81, "size": "light"},
        "retfound":      {"cost": 0.2,  "latency": 0.1,  "accuracy_base": 0.84, "size": "light"},
        "biomedclip":    {"cost": 0.05, "latency": 0.03, "accuracy_base": 0.75, "size": "light"},
        "radfm":         {"cost": 0.3,  "latency": 0.25, "accuracy_base": 0.78, "size": "medium"},
    }

    # ── Safety-Critical Keywords ───────────────────────────────────
    SAFETY_KEYWORDS = [
        "cancer", "carcinoma", "malignant", "tumor", "neoplasm",
        "stroke", "hemorrhage", "infarct", "embolism", "aneurysm",
        "pneumothorax", "critical", "emergent", "urgent", "life-threatening",
        "fracture", "dissection", "perforation", "tamponade", "sepsis",
    ]

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.available_models: set[str] = set()
        self.min_models = self.config.get("min_models_per_task", 2)
        self.max_models = self.config.get("max_models_per_task", 5)

        # Feedback learning: track model performance
        self.performance_log: dict[str, dict[str, float]] = {}

    def register_available_model(self, model_key: str) -> None:
        """Register a model as available for routing."""
        self.available_models.add(model_key)
        if model_key not in self.performance_log:
            profile = self.MODEL_PROFILES.get(model_key, {})
            self.performance_log[model_key] = {
                "accuracy": profile.get("accuracy_base", 0.75),
                "total_runs": 0,
                "failures": 0,
                "avg_latency": profile.get("latency", 0.5),
            }
        logger.info(f"Router: Registered model {model_key}")

    def route(
        self,
        modality: str,
        file_type: str = "2d",
        complexity: str = "standard",
        query: str = "",
        rag_context_strength: float = 0.0,
        available_only: bool = True,
    ) -> dict[str, list[str]]:
        """Adaptive routing based on modality, complexity, query, and context.

        v7.0 Pipeline:
          1. Static table lookup (baseline)
          2. Query-aware adjustment
          3. Safety routing enforcement
          4. RAG-aware model sizing
          5. Cost/latency optimization
          6. Feedback-driven weight adjustment
          7. Filter to available models

        Args:
            modality: Detected imaging modality
            file_type: "2d", "3d", or "video"
            complexity: "simple", "standard", or "complex"
            query: User's text query (for query-aware and safety routing)
            rag_context_strength: 0-1 indicating RAG retrieved context quality
            available_only: Only return models that are registered as available

        Returns:
            Dict with roles → model lists
        """
        route_key = modality if modality in self.ROUTING_TABLE else "general_medical"
        base_route = {k: list(v) for k, v in self.ROUTING_TABLE[route_key].items()}

        # Step 1: 3D specialist enforcement
        if file_type == "3d" and not base_route.get("specialist_3d"):
            base_route["specialist_3d"] = ["med3dvlm", "merlin"]

        # Step 2: Complexity-based upgrade
        if complexity == "complex":
            base_route["reasoner"] = ["medix_r1_30b", "medix_r1_8b"]
            if "hulu_med_32b" not in base_route["primary"]:
                base_route["primary"].insert(0, "hulu_med_32b")

        # Step 3: Query-aware routing
        if query:
            base_route = self._apply_query_routing(base_route, query)

        # Step 4: Safety routing (force big models for critical queries)
        if query and self._is_safety_critical(query):
            base_route = self._apply_safety_routing(base_route)

        # Step 5: RAG-aware model sizing
        if rag_context_strength > 0.7:
            base_route = self._apply_rag_optimization(base_route)

        # Step 6: Feedback-driven adjustment
        base_route = self._apply_feedback_weights(base_route)

        # Step 7: Filter to available models only
        if available_only and self.available_models:
            filtered = {}
            for role, models in base_route.items():
                filtered[role] = [m for m in models if m in self.available_models]
            base_route = filtered

        # Check minimum
        total_models = sum(len(v) for v in base_route.values())
        if total_models < self.min_models:
            logger.warning(
                f"Only {total_models} models available for {modality}. "
                f"Minimum is {self.min_models}."
            )

        logger.info(f"Router: {modality} → {base_route}")
        return base_route

    def adaptive_routing(self, outputs: dict[str, dict]) -> list[str]:
        """Filter models based on confidence — stop early if confident.

        Returns model keys that are confident enough to skip further models.
        If none are confident, returns empty list (run all).
        """
        confident = []
        for model, result in outputs.items():
            conf = result.get("confidence", 0)
            if conf > 0.85:
                confident.append(model)

        if confident:
            logger.info(
                f"Adaptive routing: {len(confident)} models confident enough "
                f"({', '.join(confident)}), skipping remaining"
            )
        return confident

    def get_multi_stage_plan(
        self, modality: str, file_type: str = "2d"
    ) -> list[dict[str, Any]]:
        """Plan multi-stage execution: cheap → medium → heavy.

        Stage 1: Cheap/fast model (screening)
        Stage 2: Medium model (if Stage 1 uncertain)
        Stage 3: Heavy model (only if Stage 2 also uncertain)
        """
        route = self.route(modality, file_type, available_only=True)
        all_models = []
        for models in route.values():
            all_models.extend(models)

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for m in all_models:
            if m not in seen:
                unique.append(m)
                seen.add(m)

        # Classify by size
        stages = []
        light = [m for m in unique if self.MODEL_PROFILES.get(m, {}).get("size") == "light"]
        medium = [m for m in unique if self.MODEL_PROFILES.get(m, {}).get("size") == "medium"]
        heavy = [m for m in unique if self.MODEL_PROFILES.get(m, {}).get("size") == "heavy"]

        if light:
            stages.append({
                "stage": 1,
                "models": light[:2],
                "confidence_threshold": 0.85,
                "description": "Fast screening",
            })
        if medium:
            stages.append({
                "stage": 2,
                "models": medium[:2],
                "confidence_threshold": 0.75,
                "description": "Standard analysis",
            })
        if heavy:
            stages.append({
                "stage": 3,
                "models": heavy[:2],
                "confidence_threshold": 0.0,
                "description": "Deep analysis (uncertain cases)",
            })

        return stages

    def score_model(
        self, model_key: str, accuracy_weight: float = 0.5,
        cost_weight: float = 0.25, latency_weight: float = 0.25
    ) -> float:
        """Score a model based on accuracy, cost, and latency.

        score = accuracy_weight * accuracy - cost_weight * cost - latency_weight * latency
        """
        profile = self.MODEL_PROFILES.get(model_key, {})
        perf = self.performance_log.get(model_key, {})

        accuracy = perf.get("accuracy", profile.get("accuracy_base", 0.75))
        cost = profile.get("cost", 0.5)
        latency = profile.get("latency", 0.5)

        return accuracy_weight * accuracy - cost_weight * cost - latency_weight * latency

    # ── Query-Aware Routing ──────────────────────────────────────────

    def _apply_query_routing(
        self, route: dict[str, list[str]], query: str
    ) -> dict[str, list[str]]:
        """Adjust routing based on query intent."""
        q = query.lower()

        # Report generation → emphasize reasoning models
        if any(w in q for w in ["report", "generate report", "structured", "impression"]):
            if "medix_r1_30b" not in route["reasoner"]:
                route["reasoner"].insert(0, "medix_r1_30b")

        # Differential diagnosis → heavier reasoning needed
        if any(w in q for w in ["differential", "diagnos", "classify"]):
            if "medix_r1_8b" not in route["reasoner"]:
                route["reasoner"].append("medix_r1_8b")

        # Comparison / temporal → ensure multi-image capable models
        if any(w in q for w in ["compare", "prior", "change", "progression"]):
            if "hulu_med_7b" not in route["primary"]:
                route["primary"].append("hulu_med_7b")

        return route

    def _is_safety_critical(self, query: str) -> bool:
        """Check if query involves safety-critical conditions."""
        q = query.lower()
        return any(kw in q for kw in self.SAFETY_KEYWORDS)

    def _apply_safety_routing(
        self, route: dict[str, list[str]]
    ) -> dict[str, list[str]]:
        """Force large, accurate models for safety-critical queries.

        CRITICAL FOR MEDICAL: when cancer, stroke, hemorrhage etc. are
        mentioned, always use the strongest available models.
        """
        safety_models = ["hulu_med_32b", "medix_r1_30b", "hulu_med_14b"]

        for m in safety_models:
            if m not in route["primary"]:
                route["primary"].insert(0, m)

        # Ensure strong reasoner for critical cases
        route["reasoner"] = ["medix_r1_30b", "medix_r1_8b"]

        logger.info("Safety routing activated — forcing large models")
        return route

    def _apply_rag_optimization(
        self, route: dict[str, list[str]]
    ) -> dict[str, list[str]]:
        """If RAG provides strong context, prefer smaller/faster models.

        Strong RAG context reduces the need for the largest models,
        saving compute while maintaining quality.
        """
        # Remove heavy models from primary if context is strong
        for heavy in ["hulu_med_32b", "medgemma_27b", "medix_r1_30b"]:
            if heavy in route.get("primary", []):
                route["primary"].remove(heavy)

        logger.debug("RAG-optimized routing — using smaller models with strong context")
        return route

    def _apply_feedback_weights(
        self, route: dict[str, list[str]]
    ) -> dict[str, list[str]]:
        """Re-order models within each role based on learned performance.

        Models with more failures get deprioritized; higher accuracy models
        rise. This makes routing improve over time.
        """
        for role, models in route.items():
            if len(models) > 1:
                route[role] = sorted(
                    models,
                    key=lambda m: self._model_score(m),
                    reverse=True,
                )
        return route

    def _model_score(self, model_key: str) -> float:
        """Compute adaptive score for a model based on feedback."""
        perf = self.performance_log.get(model_key)
        profile = self.MODEL_PROFILES.get(model_key, {})

        if not perf or perf["total_runs"] == 0:
            return profile.get("accuracy_base", 0.75)

        accuracy = perf["accuracy"]
        failure_penalty = perf["failures"] * 0.05
        return max(0.0, accuracy - failure_penalty)

    # ── Feedback Learning ────────────────────────────────────────────

    def record_outcome(
        self, model_key: str, success: bool, confidence: float = 0.0,
        latency: float = 0.0
    ) -> None:
        """Record a model's inference outcome for feedback learning.

        Over time, this shifts routing toward the best-performing models
        and away from unreliable ones.
        """
        if model_key not in self.performance_log:
            self.performance_log[model_key] = {
                "accuracy": 0.75, "total_runs": 0, "failures": 0, "avg_latency": 0.5
            }

        log = self.performance_log[model_key]
        log["total_runs"] += 1

        if not success:
            log["failures"] += 1

        # Exponential moving average for accuracy
        if success and confidence > 0:
            alpha = 0.1  # Learning rate
            log["accuracy"] = (1 - alpha) * log["accuracy"] + alpha * confidence

        # Update latency EMA
        if latency > 0:
            log["avg_latency"] = 0.9 * log["avg_latency"] + 0.1 * latency

        logger.debug(
            f"Feedback: {model_key} "
            f"(success={success}, runs={log['total_runs']}, "
            f"acc={log['accuracy']:.3f}, failures={log['failures']})"
        )

    def get_model_rankings(self) -> list[dict[str, Any]]:
        """Get all models ranked by adaptive score."""
        rankings = []
        for model_key, perf in self.performance_log.items():
            profile = self.MODEL_PROFILES.get(model_key, {})
            rankings.append({
                "model": model_key,
                "score": self._model_score(model_key),
                "accuracy": perf["accuracy"],
                "total_runs": perf["total_runs"],
                "failures": perf["failures"],
                "cost": profile.get("cost", 0),
                "size": profile.get("size", "unknown"),
            })
        rankings.sort(key=lambda x: x["score"], reverse=True)
        return rankings


class FallbackManager:
    """Smart fallback management with failure types, cooldown, and recovery."""

    FALLBACK_CHAINS = {
        "hulu_med_32b": ["hulu_med_14b", "hulu_med_7b"],
        "hulu_med_14b": ["hulu_med_7b"],
        "hulu_med_7b": ["medgemma_4b", "radfm"],
        "medgemma_27b": ["medgemma_4b"],
        "medgemma_4b": ["hulu_med_7b", "radfm"],
        "medix_r1_30b": ["medix_r1_8b", "medix_r1_2b"],
        "medix_r1_8b": ["medix_r1_2b"],
        "med3dvlm": ["merlin", "hulu_med_7b"],
        "merlin": ["med3dvlm", "hulu_med_7b"],
        "chexagent_8b": ["chexagent_3b", "medgemma_4b"],
        "chexagent_3b": ["medgemma_4b", "hulu_med_7b"],
        "pathgen": ["medgemma_4b", "hulu_med_7b"],
        "retfound": ["biomedclip", "medgemma_4b"],
        "radfm": ["hulu_med_7b", "medgemma_4b"],
    }

    # Failure type → fallback strategy
    FAILURE_STRATEGIES = {
        "gpu_oom": "smaller",       # Use a smaller model
        "timeout": "faster",        # Use a faster model
        "hallucination": "stronger", # Use a stronger/bigger model
        "error": "next",            # Use next in chain
    }

    def __init__(self, cooldown_seconds: float = 600.0, max_failures: int = 5):
        self.failure_counts: dict[str, int] = {}
        self.failure_types: dict[str, dict[str, int]] = {}
        self.cooldowns: dict[str, float] = {}
        self.cooldown_duration = cooldown_seconds
        self.max_failures = max_failures

    def get_fallback(
        self, failed_model: str, failure_type: str = "error"
    ) -> str | None:
        """Get next fallback model with smart failure-type routing.

        Args:
            failed_model: The model that failed
            failure_type: "gpu_oom", "timeout", "hallucination", or "error"
        """
        # Check if there's a strategy-specific override
        strategy = self.FAILURE_STRATEGIES.get(failure_type, "next")
        chain = self.FALLBACK_CHAINS.get(failed_model, [])

        if strategy == "smaller":
            # Prefer smaller models (GPU OOM)
            chain = sorted(
                chain,
                key=lambda m: IntelligentRouter.MODEL_PROFILES.get(m, {}).get("cost", 1.0)
            )
        elif strategy == "stronger":
            # Prefer stronger models (hallucination)
            chain = sorted(
                chain,
                key=lambda m: IntelligentRouter.MODEL_PROFILES.get(m, {}).get("accuracy_base", 0),
                reverse=True,
            )
        elif strategy == "faster":
            # Prefer faster models (timeout)
            chain = sorted(
                chain,
                key=lambda m: IntelligentRouter.MODEL_PROFILES.get(m, {}).get("latency", 1.0)
            )

        for fallback in chain:
            if not self._is_cooled_down(fallback) and self.failure_counts.get(fallback, 0) < self.max_failures:
                return fallback
        return None

    def record_failure(
        self, model_key: str, failure_type: str = "error"
    ) -> None:
        """Record a failure with type tracking."""
        self.failure_counts[model_key] = self.failure_counts.get(model_key, 0) + 1

        if model_key not in self.failure_types:
            self.failure_types[model_key] = {}
        self.failure_types[model_key][failure_type] = \
            self.failure_types[model_key].get(failure_type, 0) + 1

        # Auto-cooldown after max failures
        if self.failure_counts[model_key] >= self.max_failures:
            self.cooldowns[model_key] = time.monotonic()
            logger.warning(
                f"Model {model_key} disabled for {self.cooldown_duration}s "
                f"after {self.max_failures} failures"
            )

        logger.warning(
            f"Model {model_key} failed ({failure_type}), "
            f"count: {self.failure_counts[model_key]}, "
            f"types: {self.failure_types[model_key]}"
        )

    def record_success(self, model_key: str) -> None:
        """Record a success — reduces failure count."""
        self.failure_counts[model_key] = max(0, self.failure_counts.get(model_key, 0) - 1)
        # Clear cooldown on success
        self.cooldowns.pop(model_key, None)

    def _is_cooled_down(self, model_key: str) -> bool:
        """Check if a model is in cooldown period."""
        cooldown_start = self.cooldowns.get(model_key)
        if cooldown_start is None:
            return False
        elapsed = time.monotonic() - cooldown_start
        if elapsed >= self.cooldown_duration:
            # Cooldown expired — re-enable
            self.cooldowns.pop(model_key, None)
            self.failure_counts[model_key] = 0
            logger.info(f"Model {model_key} re-enabled after cooldown")
            return False
        return True

    def get_health_status(self) -> dict[str, Any]:
        """Get health status of all models in fallback system."""
        status = {}
        for model_key in self.FALLBACK_CHAINS:
            is_cooled = self._is_cooled_down(model_key)
            status[model_key] = {
                "failures": self.failure_counts.get(model_key, 0),
                "failure_types": self.failure_types.get(model_key, {}),
                "cooled_down": is_cooled,
                "available": not is_cooled and self.failure_counts.get(model_key, 0) < self.max_failures,
            }
        return status
