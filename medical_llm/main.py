"""
Medical LLM Super-Engine — Main Pipeline Orchestrator.

Full 7-layer pipeline:
    Layer 1: Input Understanding (NER, negation, symptoms, classification)
    Layer 2: Smart Router (query → model ensemble selection)
    Layer 3: Multi-Model Core (DeepSeek reasoning, medical models, RAG)
    Layer 4: Meta-Fusion (consensus, confidence scoring)
    Layer 5: Safety & Validation (hallucination, drugs, risks)
    Layer 6: Response Generation (structured reports)
    Layer 7: Orchestration (this file)

Usage:
    from main import MedicalLLMEngine
    engine = MedicalLLMEngine()
    engine.initialize()
    result = engine.analyze("Patient presents with severe chest pain...")
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import threading
import time
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

import yaml

# ── Layer 1: Input Understanding ───────────────────────────
from core.input.medical_ner import MedicalNER
from core.input.negation_detector import NegationDetector
from core.input.symptom_extractor import SymptomExtractor
from core.input.query_classifier import QueryClassifier

# ── Layer 2: Smart Router ──────────────────────────────────
from core.routing.smart_router import SmartRouter, FallbackManager

# ── Layer 3: Model Engines ─────────────────────────────────
from core.models.base_model import BaseLLM, HuggingFaceLLM, VLLMEngine
from core.models.reasoning.deepseek_engine import DeepSeekEngine
from core.models.medical.meditron_engine import MeditronEngine
from core.models.medical.mellama_engine import MeLLaMAEngine
from core.models.medical.pmc_llama_engine import PMCLLaMAEngine
from core.models.medical.openbiollm_engine import OpenBioLLMEngine
from core.models.clinical.biomistral_engine import BioMistralEngine
from core.models.clinical.clinical_camel_engine import ClinicalCamelEngine
from core.models.clinical.med42_engine import Med42Engine
from core.models.conversational.chatdoctor_engine import ChatDoctorEngine

# ── RAG ────────────────────────────────────────────────────
from core.rag.medical_rag import MedicalRAG
from core.rag.pubmed_fetcher import PubMedFetcher
from core.rag.web_search import WebSearch
from core.rag.knowledge_base import KnowledgeBase
from core.rag.retrieval_pipeline import MedicalRetrievalPipeline

# ── Layer 4: Fusion ────────────────────────────────────────
from core.fusion.meta_fusion import MetaFusion, UncertaintyEstimator, ContradictionDetector

# ── Layer 5: Safety ────────────────────────────────────────
from core.safety.safety import (
    HallucinationDetector, DrugInteractionChecker,
    ClinicalValidator, RiskFlagger,
)

# ── Layer 6: Response ──────────────────────────────────────
from core.response.report_generator import ReportGenerator, ResponseStyler

# ── Layer 7: Execution & Governance ────────────────────────
from core.execution.parallel_executor import ParallelExecutor
from core.governance.audit import AuditLogger


logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    """Parse a boolean environment variable safely."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    """Parse an integer environment variable with sane fallbacks."""
    raw = os.getenv(name)
    if raw is None:
        return max(minimum, default)
    try:
        return max(minimum, int(raw))
    except ValueError:
        logger.warning("Invalid integer env var %s=%r; using default %s", name, raw, default)
        return max(minimum, default)

# ── Model registry mapping config keys → engine classes ────
MODEL_REGISTRY: dict[str, type[BaseLLM]] = {
    "deepseek_r1": DeepSeekEngine,
    "meditron_70b": MeditronEngine,
    "mellama_13b": MeLLaMAEngine,
    "pmc_llama_13b": PMCLLaMAEngine,
    "openbiollm_70b": OpenBioLLMEngine,
    "biomistral_7b": BioMistralEngine,
    "clinical_camel_70b": ClinicalCamelEngine,
    "med42_70b": Med42Engine,
    "chatdoctor": ChatDoctorEngine,
}


class MedicalLLMEngine:
    """Main orchestrator — ties all 7 layers together.

    Initializes all components from YAML config, loads models on demand,
    and runs the full pipeline for each medical query.
    """

    # LRU cache for repeated queries
    _CACHE_MAX_SIZE = 128
    _CONVERSATION_HISTORY_MAX_TURNS = 20
    _MAX_CONVERSATION_SESSIONS = 128
    _SHARED_STATE_RETRY_SECONDS = 30.0

    # Input sanitization patterns
    _INJECTION_PATTERNS = [
        re.compile(r'\{\{.*?\}\}'),       # Template injection
        re.compile(r'<\|.*?\|>'),           # Model control tokens
        re.compile(r'\[INST\]|\[/INST\]'),  # Instruction injection
        re.compile(r'</?s>'),               # BOS/EOS injection
    ]

    def __init__(
        self,
        model_config_path: str = "config/model_config.yaml",
        pipeline_config_path: str = "config/pipeline_config.yaml",
        redis_url: str | None = None,
        cache_ttl_seconds: int | None = None,
        session_ttl_seconds: int | None = None,
        enable_shared_state: bool | None = None,
        shared_state_prefix: str | None = None,
    ):
        self.model_config_path = Path(model_config_path)
        self.pipeline_config_path = Path(pipeline_config_path)

        self.model_config: dict[str, Any] = {}
        self.pipeline_config: dict[str, Any] = {}

        # Component instances (set in initialize())
        self._models: dict[str, BaseLLM] = {}
        self._ner: MedicalNER | None = None
        self._negation: NegationDetector | None = None
        self._symptoms: SymptomExtractor | None = None
        self._classifier: QueryClassifier | None = None
        self._router: SmartRouter | None = None
        self._fallback: FallbackManager | None = None
        self._rag: MedicalRAG | None = None
        self._pubmed: PubMedFetcher | None = None
        self._web_search: WebSearch | None = None
        self._knowledge_base: KnowledgeBase | None = None
        self._retrieval_pipeline: MedicalRetrievalPipeline | None = None
        self._fusion: MetaFusion | None = None
        self._uncertainty: UncertaintyEstimator | None = None
        self._contradictions: ContradictionDetector | None = None
        self._hallucination: HallucinationDetector | None = None
        self._drug_checker: DrugInteractionChecker | None = None
        self._clinical_validator: ClinicalValidator | None = None
        self._risk_flagger: RiskFlagger | None = None
        self._report_generator: ReportGenerator | None = None
        self._response_styler: ResponseStyler | None = None
        self._executor: ParallelExecutor | None = None
        self._audit: AuditLogger | None = None
        self._is_initialized = False

        # Query result cache (LRU)
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()

        # Session-scoped state for interactive/API use
        self._conversation_histories: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
        self._conversation_timestamps: dict[str, float] = {}
        self._state_lock = threading.RLock()
        self._redis_url = redis_url or os.getenv("MEDICAL_LLM_REDIS_URL") or os.getenv("REDIS_URL")
        self._shared_state_enabled = (
            _env_bool("MEDICAL_LLM_SHARED_STATE_ENABLED", True)
            if enable_shared_state is None
            else enable_shared_state
        )
        self._shared_state_prefix = (
            shared_state_prefix
            or os.getenv("MEDICAL_LLM_STATE_KEY_PREFIX")
            or "medai:medical-llm"
        ).rstrip(":")
        self._cache_ttl_seconds = (
            _env_int("MEDICAL_LLM_CACHE_TTL_SECONDS", 300, minimum=30)
            if cache_ttl_seconds is None
            else max(30, cache_ttl_seconds)
        )
        self._SESSION_MAX_AGE_SECONDS = (
            _env_int("MEDICAL_LLM_SESSION_TTL_SECONDS", 3600, minimum=300)
            if session_ttl_seconds is None
            else max(300, session_ttl_seconds)
        )
        self._shared_state_client: Any | None = None
        self._shared_state_available = False
        self._last_shared_state_retry_at = 0.0

        # Runtime model residency controls for GPU-constrained deployments.
        self._model_state_lock = threading.RLock()
        self._resident_models: OrderedDict[str, None] = OrderedDict()
        self._active_model_calls: dict[str, int] = {}
        self._max_resident_models = 0
        self._auto_unload_after_inference = False
        self._sequential_heavy_models = False

    def initialize(self, models_to_load: list[str] | None = None) -> None:
        """Initialize the full engine from config files.

        Args:
            models_to_load: Optional list of model keys to load.
                If None, registers all models but loads on-demand.
        """
        logger.info("═" * 60)
        logger.info("  🏥 Medical LLM Super-Engine — Initializing")
        logger.info("═" * 60)

        # Load configs
        self._load_configs()

        # Layer 1: Input Understanding
        self._init_input_layer()

        # Layer 2: Smart Router
        self._init_router()

        # RAG Engine
        self._init_rag()

        # Layer 3: Register Models
        self._init_models(models_to_load)

        # Layer 4: Fusion
        self._init_fusion()

        # Layer 5: Safety
        self._init_safety()

        # Layer 6: Response
        self._init_response()

        # Layer 7: Execution & Governance
        self._init_execution()
        self._ensure_shared_state_client()

        self._is_initialized = True
        state_backend = self.get_state_backend_status()
        logger.info("═" * 60)
        logger.info("  ✅ Engine initialized successfully")
        logger.info(f"  📊 Models registered: {len(self._models)}")
        logger.info(f"  📚 RAG vectors: {self._rag.get_stats()['total_vectors'] if self._rag else 0}")
        logger.info("  🗃️  Shared state backend: %s", state_backend["mode"])
        logger.info("═" * 60)

    def _load_configs(self) -> None:
        """Load YAML configuration files."""
        if self.model_config_path.exists():
            with open(self.model_config_path, "r", encoding="utf-8") as f:
                self.model_config = yaml.safe_load(f) or {}
            logger.info(f"Model config loaded: {self.model_config_path}")
        else:
            logger.warning(f"Model config not found: {self.model_config_path}")

        if self.pipeline_config_path.exists():
            with open(self.pipeline_config_path, "r", encoding="utf-8") as f:
                self.pipeline_config = yaml.safe_load(f) or {}
            logger.info(f"Pipeline config loaded: {self.pipeline_config_path}")
        else:
            logger.warning(f"Pipeline config not found: {self.pipeline_config_path}")

    def _init_input_layer(self) -> None:
        """Initialize Layer 1 components."""
        self._ner = MedicalNER()
        self._negation = NegationDetector()
        self._symptoms = SymptomExtractor()
        self._classifier = QueryClassifier()
        logger.info("Layer 1 (Input Understanding) initialized")

    def _init_router(self) -> None:
        """Initialize Layer 2 router."""
        routing_config = self.pipeline_config.get("routing", {})
        self._router = SmartRouter(config=routing_config)
        self._fallback = FallbackManager()
        logger.info("Layer 2 (Smart Router) initialized")

    def _init_rag(self) -> None:
        """Initialize RAG engine."""
        rag_config = self.pipeline_config.get("rag", {})
        vector_backend = os.getenv(
            "MEDICAL_LLM_VECTOR_STORE",
            rag_config.get("vector_store", "faiss"),
        )

        embedding_name = self.model_config.get("embedding", {}).get(
            "model_id", "BAAI/bge-large-en-v1.5"
        )
        self._rag = MedicalRAG(
            embedding_model=embedding_name,
            persist_dir=rag_config.get("persist_dir", "./data/rag/faiss_index"),
            chunk_size=rag_config.get("chunk_size", 512),
            chunk_overlap=rag_config.get("chunk_overlap", 64),
            vector_backend=vector_backend,
            qdrant_url=os.getenv("MEDICAL_LLM_QDRANT_URL", rag_config.get("qdrant_url", "")),
            qdrant_collection=rag_config.get("qdrant_collection", "medical_rag"),
            qdrant_api_key=os.getenv(
                "MEDICAL_LLM_QDRANT_API_KEY",
                rag_config.get("qdrant_api_key", ""),
            ),
        )

        if rag_config.get("enable_pubmed", True):
            pubmed_config = rag_config.get("pubmed", {})
            self._pubmed = PubMedFetcher(
                max_results=pubmed_config.get("max_results", 10),
                email=pubmed_config.get("email", "medllm@research.org"),
                tool_name=pubmed_config.get("tool_name", "MedicalLLMEngine"),
                cache_dir=pubmed_config.get("cache_dir", "./data/rag/pubmed_cache"),
                api_key=os.getenv(
                    "MEDICAL_LLM_PUBMED_API_KEY",
                    os.getenv("PUBMED_API_KEY", pubmed_config.get("api_key", "")),
                ),
            )
        else:
            self._pubmed = None

        if rag_config.get("enable_web_search", True):
            web_config = rag_config.get("web_search", {})
            self._web_search = WebSearch(config=web_config)
        else:
            self._web_search = None

        self._knowledge_base = KnowledgeBase()
        self._retrieval_pipeline = MedicalRetrievalPipeline(
            rag_engine=self._rag,
            pubmed_fetcher=self._pubmed,
            web_search=self._web_search,
            config=rag_config.get("search_architecture", {}),
        )
        logger.info("RAG engine initialized")

    def _init_models(self, models_to_load: list[str] | None = None) -> None:
        """Register and optionally load model engines."""
        models_section = self.model_config.get("models", {})
        default_model_config = self.model_config.get("defaults", {})

        for model_key, model_conf in models_section.items():
            if not model_conf.get("enabled", True):
                continue

            engine_cls = MODEL_REGISTRY.get(model_key)
            if engine_cls is None:
                logger.warning(f"No engine class for model key: {model_key}")
                continue

            merged_config = {**default_model_config, **model_conf}
            model_id = merged_config.get("model_id", "")
            config = {
                **merged_config,
                "key": model_key,
                "role": merged_config.get("role", "primary"),
                "weight": merged_config.get("weight", 0.5),
                "device_map": merged_config.get("device_map", "auto"),
                "torch_dtype": merged_config.get(
                    "torch_dtype",
                    merged_config.get("dtype", "bfloat16"),
                ),
                "trust_remote_code": merged_config.get("trust_remote_code", False),
                # DeepSeek-specific
                "self_reflection_passes": merged_config.get("self_reflection_passes", 3),
                "constraint_prompting": merged_config.get("constraint_prompting", True),
            }

            engine = engine_cls(model_id=model_id, config=config)
            self._models[model_key] = engine
            self._router.register_model(model_key)

            logger.info(f"  Registered: {model_key} → {model_id}")

        # Load specified models immediately
        if models_to_load:
            for key in models_to_load:
                if key in self._models:
                    self._models[key].load()

    def _init_fusion(self) -> None:
        """Initialize Layer 4 fusion."""
        fusion_config = self.pipeline_config.get("fusion", {})
        self._fusion = MetaFusion(config=fusion_config)
        self._uncertainty = UncertaintyEstimator()
        self._contradictions = ContradictionDetector()
        logger.info("Layer 4 (Meta-Fusion) initialized")

    def _init_safety(self) -> None:
        """Initialize Layer 5 safety."""
        self._hallucination = HallucinationDetector(
            rag_engine=self._rag,
            web_search=self._web_search,
        )
        self._drug_checker = DrugInteractionChecker()
        self._clinical_validator = ClinicalValidator()
        self._risk_flagger = RiskFlagger()
        logger.info("Layer 5 (Safety & Validation) initialized")

    def _init_response(self) -> None:
        """Initialize Layer 6 response."""
        self._report_generator = ReportGenerator()
        self._response_styler = ResponseStyler()
        logger.info("Layer 6 (Response Generator) initialized")

    def _init_execution(self) -> None:
        """Initialize Layer 7 execution."""
        exec_config = self.pipeline_config.get("execution", {})
        self._executor = ParallelExecutor(
            max_workers=exec_config.get("max_concurrent_models", 4),
            timeout=exec_config.get("model_timeout_seconds", 300),
            retry_on_failure=exec_config.get("retry_on_failure", False),
            max_retries=exec_config.get("max_retries", 2),
        )
        governance_config = self.pipeline_config.get("governance", {})
        audit_config = governance_config.get("audit", {})
        self._audit = AuditLogger(
            log_dir=audit_config.get("log_dir", "./logs/audit"),
            enabled=audit_config.get("enabled", True),
            enable_detailed=audit_config.get("enable_detailed", True),
            enable_console=audit_config.get("enable_console", False),
        )
        lazy_loading = exec_config.get("lazy_loading", {})
        self._max_resident_models = _env_int(
            "MEDICAL_LLM_MAX_RESIDENT_MODELS",
            int(lazy_loading.get("max_resident_models", 0)),
            minimum=0,
        )
        self._auto_unload_after_inference = _env_bool(
            "MEDICAL_LLM_AUTO_UNLOAD_AFTER_REQUEST",
            bool(lazy_loading.get("auto_unload_after_inference", False)),
        )
        self._sequential_heavy_models = _env_bool(
            "MEDICAL_LLM_SEQUENTIAL_HEAVY_MODELS",
            bool(lazy_loading.get("sequential_heavy_models", False)),
        )
        logger.info("Layer 7 (Execution & Governance) initialized")
        logger.info(
            "Model residency policy: max_resident=%d auto_unload=%s sequential_heavy=%s",
            self._max_resident_models,
            self._auto_unload_after_inference,
            self._sequential_heavy_models,
        )

    def _state_key(self, namespace: str, identifier: str) -> str:
        """Build a namespaced shared-state key."""
        return f"{self._shared_state_prefix}:{namespace}:{identifier}"

    def _ensure_shared_state_client(self) -> Any | None:
        """Lazily connect to Redis for shared cache and conversation state."""
        if not self._shared_state_enabled or not self._redis_url:
            return None

        now = time.monotonic()
        with self._state_lock:
            if self._shared_state_client is not None:
                return self._shared_state_client
            if now - self._last_shared_state_retry_at < self._SHARED_STATE_RETRY_SECONDS:
                return None
            self._last_shared_state_retry_at = now

        try:
            import redis

            client = redis.Redis.from_url(
                self._redis_url,
                decode_responses=True,
                socket_timeout=2.0,
                socket_connect_timeout=2.0,
                retry_on_timeout=True,
                health_check_interval=30,
            )
            client.ping()
        except Exception as exc:
            with self._state_lock:
                self._shared_state_client = None
                self._shared_state_available = False
            logger.warning("Shared state backend unavailable; using memory fallback: %s", exc)
            return None

        with self._state_lock:
            self._shared_state_client = client
            self._shared_state_available = True
        logger.info("Shared Redis state backend connected for Medical LLM Engine")
        return client

    def _mark_shared_state_unavailable(self, action: str, exc: Exception) -> None:
        """Degrade gracefully to memory-only mode after a shared-state failure."""
        with self._state_lock:
            self._shared_state_client = None
            self._shared_state_available = False
        logger.warning("Shared state %s failed; falling back to memory: %s", action, exc)

    def get_state_backend_status(self) -> dict[str, Any]:
        """Return the current shared-state backend status."""
        client = self._ensure_shared_state_client()
        with self._state_lock:
            mode = "redis" if client is not None else "memory"
            return {
                "mode": mode,
                "redis_configured": bool(self._redis_url),
                "shared_state_enabled": self._shared_state_enabled,
                "cache_ttl_seconds": self._cache_ttl_seconds,
                "session_ttl_seconds": self._SESSION_MAX_AGE_SECONDS,
            }

    def _cache_result_locally(self, cache_key: str, result: dict[str, Any]) -> None:
        """Store a defensive copy of a result in the local LRU cache."""
        with self._state_lock:
            self._cache[cache_key] = deepcopy(result)
            self._cache.move_to_end(cache_key)
            if len(self._cache) > self._CACHE_MAX_SIZE:
                self._cache.popitem(last=False)

    def _get_cached_result(self, cache_key: str) -> dict[str, Any] | None:
        """Read a cached analysis result from Redis or local memory."""
        client = self._ensure_shared_state_client()
        if client is not None:
            try:
                raw = client.get(self._state_key("cache", cache_key))
                if raw:
                    result = json.loads(raw)
                    self._cache_result_locally(cache_key, result)
                    return result
                with self._state_lock:
                    self._cache.pop(cache_key, None)
                return None
            except Exception as exc:
                self._mark_shared_state_unavailable("cache read", exc)

        with self._state_lock:
            cached = self._cache.get(cache_key)
            if cached is None:
                return None
            self._cache.move_to_end(cache_key)
            return deepcopy(cached)

    def _set_cached_result(self, cache_key: str, result: dict[str, Any]) -> None:
        """Persist a cacheable result to local memory and shared state."""
        self._cache_result_locally(cache_key, result)

        client = self._ensure_shared_state_client()
        if client is None:
            return
        try:
            client.setex(
                self._state_key("cache", cache_key),
                self._cache_ttl_seconds,
                json.dumps(result, default=str),
            )
        except Exception as exc:
            self._mark_shared_state_unavailable("cache write", exc)

    def _prune_memory_sessions(self) -> None:
        """Prune local in-memory conversation sessions by age and size."""
        if len(self._conversation_histories) > self._MAX_CONVERSATION_SESSIONS:
            while len(self._conversation_histories) > self._MAX_CONVERSATION_SESSIONS:
                evicted_id, _ = self._conversation_histories.popitem(last=False)
                self._conversation_timestamps.pop(evicted_id, None)

        now = time.time()
        stale_ids = [
            sid for sid, ts in self._conversation_timestamps.items()
            if now - ts > self._SESSION_MAX_AGE_SECONDS
        ]
        for sid in stale_ids:
            self._conversation_histories.pop(sid, None)
            self._conversation_timestamps.pop(sid, None)
        if stale_ids:
            logger.info("Evicted %d stale in-memory conversation session(s)", len(stale_ids))

    def _set_memory_conversation_history(
        self,
        session_id: str,
        history: list[dict[str, str]],
    ) -> None:
        """Replace a local in-memory conversation history snapshot."""
        trimmed_history = [dict(turn) for turn in history[-self._CONVERSATION_HISTORY_MAX_TURNS:]]
        with self._state_lock:
            self._conversation_histories[session_id] = trimmed_history
            self._conversation_histories.move_to_end(session_id)
            self._conversation_timestamps[session_id] = time.time()
            self._prune_memory_sessions()

    def _append_memory_conversation_history(
        self,
        session_id: str,
        turns: list[dict[str, str]],
    ) -> None:
        """Append new turns to the local in-memory conversation history."""
        with self._state_lock:
            history = self._conversation_histories.get(session_id, [])
            history = history + [dict(turn) for turn in turns]
            history = history[-self._CONVERSATION_HISTORY_MAX_TURNS:]
            self._conversation_histories[session_id] = history
            self._conversation_histories.move_to_end(session_id)
            self._conversation_timestamps[session_id] = time.time()
            self._prune_memory_sessions()

    def _get_memory_conversation_history(self, session_id: str) -> list[dict[str, str]]:
        """Read a defensive copy of the local in-memory conversation history."""
        with self._state_lock:
            self._prune_memory_sessions()
            history = self._conversation_histories.get(session_id, [])
            if session_id in self._conversation_histories:
                self._conversation_histories.move_to_end(session_id)
                self._conversation_timestamps[session_id] = time.time()
            return [dict(turn) for turn in history]

    # ═══════════════════════════════════════════════════════════
    #  MAIN PIPELINE
    # ═══════════════════════════════════════════════════════════

    def analyze(
        self,
        query: str,
        mode: str = "doctor",
        enable_rag: bool = True,
        force_models: list[str] | None = None,
        use_cache: bool = True,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Run the full 7-layer analysis pipeline.

        Args:
            query: Medical query text
            mode: Audience mode (doctor/patient/research)
            enable_rag: Whether to use RAG retrieval
            force_models: Override router with specific models
            use_cache: Enable query result caching
            session_id: Optional session identifier for multi-turn conversation state

        Returns:
            Complete analysis result with report, metadata, and audit trail
        """
        if not self._is_initialized:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        # SEC-1: Sanitize input
        query = self._sanitize_input(query)
        if not query:
            return {"error": "Empty query after sanitization", "report_text": ""}

        cache_allowed = use_cache and not session_id
        cache_key = self._get_cache_key(query, mode, enable_rag, force_models)
        if cache_allowed:
            cached_result = self._get_cached_result(cache_key)
            if cached_result is not None:
                logger.info("Cache hit — returning cached result")
                return cached_result

        conversation_history = self._get_conversation_history(session_id)

        start_time = time.time()
        logger.info(f"\n{'═' * 60}")
        logger.info("  📥 New Query received (length=%d)", len(query))
        logger.info(f"{'═' * 60}")

        # ── LAYER 1: Input Understanding ───────────────────
        logger.info("▶ Layer 1: Input Understanding")
        input_analysis = self._layer1_input(query)

        # CRISIS: Immediate safety response for self-harm/suicidal content
        if input_analysis.get("is_crisis"):
            logger.warning("🚨 CRISIS DETECTED — returning immediate safety response")
            crisis_report = self._crisis_response(query)
            return crisis_report

        # UX-4: Emergency fast-path escalation
        if input_analysis.get("is_emergency"):
            logger.warning("🚨 EMERGENCY DETECTED — escalating to maximum model ensemble")
            force_models = None  # Let router use emergency route
            input_analysis["query_category"] = "emergency"
            enable_rag = True

        # ── LAYER 2: Smart Routing ─────────────────────────
        logger.info("▶ Layer 2: Smart Routing")
        route = self._layer2_route(input_analysis, enable_rag, force_models)

        # ── RAG Retrieval ──────────────────────────────────
        rag_context = ""
        rag_evidence = []
        retrieval_info: dict[str, Any] = {}
        if route.get("enable_rag"):
            logger.info("▶ RAG: Retrieving evidence")
            rag_context, rag_evidence, retrieval_info = self._retrieve_evidence(query)

        # ── LAYER 3: Multi-Model Inference ─────────────────
        logger.info("▶ Layer 3: Multi-Model Inference")
        model_results = self._layer3_inference(
            query, route, rag_context, conversation_history
        )

        # ── LAYER 4: Meta-Fusion ───────────────────────────
        logger.info("▶ Layer 4: Meta-Fusion")
        fused = self._layer4_fusion(model_results, rag_evidence)

        # ── LAYER 5: Safety & Validation ───────────────────
        logger.info("▶ Layer 5: Safety & Validation")
        safety_result = self._layer5_safety(
            query, fused, input_analysis, rag_evidence
        )

        # ── LAYER 6: Response Generation ───────────────────
        logger.info("▶ Layer 6: Response Generation")
        report = self._layer6_response(
            fused,
            safety_result,
            rag_evidence,
            mode,
            retrieval_sources=retrieval_info.get("sources", []),
        )

        # ── LAYER 7: Audit & Finalization ──────────────────
        execution_time = time.time() - start_time
        logger.info("▶ Layer 7: Audit & Finalization")
        audit_id = self._layer7_audit(
            query, input_analysis, route, model_results,
            fused, safety_result, report, execution_time,
        )

        self._update_conversation_history(
            session_id=session_id,
            query=query,
            answer=fused.get("consensus_answer", "")[:500],
        )

        logger.info(f"{'═' * 60}")
        logger.info(f"  ✅ Analysis complete in {execution_time:.2f}s")
        logger.info(f"  📊 Confidence: {fused.get('confidence', 0):.3f}")
        logger.info(f"  ⚠️  Risk: {safety_result.get('risk_level', 'unknown')}")
        logger.info(f"  📝 Audit ID: {audit_id}")
        logger.info(f"{'═' * 60}")

        result = {
            "report": report,
            "report_text": report.get("report_text", ""),
            "confidence": fused.get("confidence", 0),
            "agreement": fused.get("agreement_score", 0),
            "risk_level": safety_result.get("risk_level", "unknown"),
            "audit_id": audit_id,
            "execution_time": round(execution_time, 3),
            "input_analysis": input_analysis,
            "routing": route,
            "model_results": model_results,
            "fusion": fused,
            "safety": safety_result,
            "rag_evidence": rag_evidence,
            "sources": retrieval_info.get("sources", []),
            "retrieval": retrieval_info,
        }

        # UX-6: Store in cache
        if cache_allowed:
            self._set_cached_result(cache_key, result)

        return result

    def _sanitize_input(self, query: str) -> str:
        """SEC-1: Sanitize user input to prevent prompt injection."""
        sanitized = query.strip()
        for pattern in self._INJECTION_PATTERNS:
            sanitized = pattern.sub('', sanitized)
        # Limit query length
        if len(sanitized) > 10000:
            sanitized = sanitized[:10000]
            logger.warning("Query truncated to 10000 characters")
        return sanitized

    def _get_cache_key(self, query: str, mode: str, enable_rag: bool = True, force_models: list | None = None) -> str:
        """Generate deterministic cache key including all parameters that affect output."""
        models_str = ",".join(sorted(force_models)) if force_models else ""
        return hashlib.sha256(f"{query}::{mode}::{enable_rag}::{models_str}".encode()).hexdigest()[:32]

    def _get_conversation_history(self, session_id: str | None) -> list[dict[str, str]]:
        """Return a defensive copy of the session conversation history."""
        if not session_id:
            return []

        client = self._ensure_shared_state_client()
        if client is not None:
            try:
                raw_turns = client.lrange(self._state_key("session", session_id), 0, -1)
                if raw_turns:
                    history = [json.loads(turn) for turn in raw_turns]
                    self._set_memory_conversation_history(session_id, history)
                    return [dict(turn) for turn in history]
                self.clear_conversation(session_id)
                return []
            except Exception as exc:
                self._mark_shared_state_unavailable("conversation read", exc)

        return self._get_memory_conversation_history(session_id)

    def _update_conversation_history(
        self,
        session_id: str | None,
        query: str,
        answer: str,
    ) -> None:
        """Persist bounded conversation history for a named session."""
        if not session_id:
            return

        turns = [
            {"role": "user", "content": query},
            {"role": "assistant", "content": answer},
        ]
        self._append_memory_conversation_history(session_id, turns)

        client = self._ensure_shared_state_client()
        if client is None:
            return
        try:
            encoded_turns = [json.dumps(turn, default=str) for turn in turns]
            session_key = self._state_key("session", session_id)
            pipeline = client.pipeline()
            pipeline.rpush(session_key, *encoded_turns)
            pipeline.ltrim(session_key, -self._CONVERSATION_HISTORY_MAX_TURNS, -1)
            pipeline.expire(session_key, self._SESSION_MAX_AGE_SECONDS)
            pipeline.execute()
        except Exception as exc:
            self._mark_shared_state_unavailable("conversation write", exc)

    def clear_conversation(self, session_id: str) -> bool:
        """Remove stored conversation history for a session."""
        shared_cleared = False
        client = self._ensure_shared_state_client()
        if client is not None:
            try:
                shared_cleared = bool(client.delete(self._state_key("session", session_id)))
            except Exception as exc:
                self._mark_shared_state_unavailable("conversation delete", exc)

        with self._state_lock:
            self._conversation_timestamps.pop(session_id, None)
            local_cleared = self._conversation_histories.pop(session_id, None) is not None
        return shared_cleared or local_cleared

    def get_conversation_session_count(self) -> int:
        """Return number of session histories currently retained."""
        client = self._ensure_shared_state_client()
        if client is not None:
            try:
                cursor = 0
                count = 0
                pattern = self._state_key("session", "*")
                while True:
                    cursor, keys = client.scan(cursor=cursor, match=pattern, count=100)
                    count += len(keys)
                    if cursor == 0:
                        return count
            except Exception as exc:
                self._mark_shared_state_unavailable("conversation count", exc)

        with self._state_lock:
            self._prune_memory_sessions()
            return len(self._conversation_histories)

    def _crisis_response(self, query: str) -> dict[str, Any]:
        """Return an immediate safety response for crisis/self-harm queries.

        This bypasses the full pipeline to provide immediate help resources.
        """
        crisis_text = (
            "I'm concerned about what you've shared. Your safety matters.\n\n"
            "PLEASE REACH OUT FOR IMMEDIATE HELP:\n\n"
            "  - National Suicide Prevention Lifeline: 988 (call or text)\n"
            "  - Crisis Text Line: Text HOME to 741741\n"
            "  - International Association for Suicide Prevention: "
            "https://www.iasp.info/resources/Crisis_Centres/\n"
            "  - Emergency Services: 911 (US) / 999 (UK) / 112 (EU)\n\n"
            "You are not alone. Trained counselors are available 24/7 and "
            "want to help.\n\n"
            "This AI system is NOT equipped to provide crisis counseling. "
            "Please contact one of the resources above immediately."
        )
        return {
            "report": {"report_text": crisis_text},
            "report_text": crisis_text,
            "confidence": 1.0,
            "agreement": 1.0,
            "risk_level": "emergent",
            "audit_id": "crisis_response",
            "execution_time": 0.0,
            "input_analysis": {"query_category": "emergency", "is_crisis": True},
            "routing": {},
            "model_results": [],
            "fusion": {},
            "safety": {"risk_level": "emergent", "is_crisis": True},
            "rag_evidence": [],
        }

    def _layer1_input(self, query: str) -> dict[str, Any]:
        """Layer 1: Analyze input query."""
        classification = self._classifier.classify(query)
        symptoms = self._symptoms.extract_all(query)
        medications = self._symptoms.extract_medications(query)

        # BUG-2 FIX: Actually use NER for entity extraction
        ner_entities = []
        try:
            ner_entities = self._ner.extract(query)
        except Exception as e:
            logger.warning(f"NER extraction failed (non-fatal): {e}")

        # Extract entities & check negation
        entities = []
        negated_entities = []
        if symptoms["symptoms"]:
            symptom_texts = [s["symptom"] for s in symptoms["symptoms"]]
            affirmed, negated = self._negation.filter_entities(query, symptom_texts)
            entities = affirmed
            negated_entities = negated

        return {
            "query_category": classification["category"],
            "query_confidence": classification["confidence"],
            "is_emergency": classification.get("is_emergency", False),
            "is_crisis": classification.get("is_crisis", False),
            "symptoms": symptoms,
            "medications": medications,
            "entities": entities,
            "negated_entities": negated_entities,
            "ner_entities": ner_entities,
        }

    def _layer2_route(
        self,
        input_analysis: dict[str, Any],
        enable_rag: bool,
        force_models: list[str] | None,
    ) -> dict[str, Any]:
        """Layer 2: Determine routing."""
        category = input_analysis["query_category"]

        # Estimate complexity
        symptom_count = input_analysis["symptoms"].get("symptom_count", 0)
        if symptom_count >= 5 or category == "emergency":
            complexity = "complex"
        elif symptom_count >= 2 or category in ("diagnosis", "differential"):
            complexity = "standard"
        else:
            complexity = "simple"

        route = self._router.route(
            query_category=category,
            complexity=complexity,
            enable_rag=enable_rag,
            force_models=force_models,
        )

        return route

    def _retrieve_evidence(self, query: str) -> tuple[str, list[dict], dict[str, Any]]:
        """Retrieve evidence through the full browsing-style RAG pipeline."""
        if self._retrieval_pipeline is not None:
            try:
                max_results = int(self.pipeline_config.get("rag", {}).get("top_k", 7))
                retrieval = self._retrieval_pipeline.retrieve(
                    query,
                    max_results=max_results,
                )
                return retrieval.context, retrieval.rag_evidence, retrieval.to_dict()
            except Exception as exc:
                logger.warning("Advanced retrieval pipeline failed, using legacy fallback: %s", exc)

        rag_evidence: list[dict[str, Any]] = []
        context_parts: list[str] = []
        sources: list[dict[str, Any]] = []

        # Legacy fallback: vector search
        if self._rag:
            try:
                self._rag.initialize()
                results = self._rag.query(query, top_k=5)
                rag_evidence.extend(results)
                if results:
                    context_parts.append("### Knowledge Base Evidence")
                    for r in results[:3]:
                        source = r.get("metadata", {}).get("source", "vector_db")
                        context_parts.append(
                            f"- (relevance: {r['relevance']:.3f}) {r['content'][:300]}"
                        )
                        sources.append({
                            "title": r.get("metadata", {}).get("title", "Knowledge Base"),
                            "url": source if str(source).startswith(("http://", "https://")) else "",
                            "source": "vector_db",
                            "type": "vector_db",
                            "confidence": round(float(r.get("relevance", 0.0)), 3),
                        })
            except Exception as e:
                logger.warning(f"RAG query failed: {e}")

        context = "\n".join(context_parts)
        retrieval = {
            "query_analysis": {
                "original_query": query,
                "sanitized_query": query,
                "redacted_query": query,
                "intent": "legacy_fallback",
                "needs_search": False,
                "needs_freshness": False,
                "medical_only": True,
                "removed_phi": [],
                "rewritten_queries": [query],
                "selected_sources": ["vector_db"],
            },
            "context": context,
            "sources": sources,
            "warnings": [],
            "metadata": {
                "sources_queried": ["vector_db"] if sources else [],
                "candidate_documents": len(rag_evidence),
                "retained_documents": len(rag_evidence),
                "cached": False,
                "orchestrator": "legacy_fallback",
            },
            "documents": sources,
        }
        return context, rag_evidence, retrieval

    def _layer3_inference(
        self,
        query: str,
        route: dict[str, Any],
        rag_context: str,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> list[dict[str, Any]]:
        """Layer 3: Run model inference.

        Two-phase execution:
          Phase 1: Run primary/medical/reasoning models in parallel
          Phase 2: Run verifiers against actual Phase 1 output
        """
        # ── Phase 1: Primary inference ──────────────────────────
        primary_tasks = []
        verifier_keys: list[tuple[str, str]] = []

        for role in ["primary", "medical", "reasoning", "verifier"]:
            for key in route.get(role, []):
                if role == "verifier":
                    verifier_keys.append((key, role))
                else:
                    engine = self._models.get(key)
                    if engine is None:
                        continue
                    fn, kwargs = self._dispatch_engine(
                        engine, key, role, query, rag_context, route, conversation_history
                    )
                    primary_tasks.append({
                        "name": f"{key}::{role}",
                        "fn": self._managed_model_call,
                        "kwargs": {
                            "model_key": key,
                            "callable_fn": fn,
                            "call_kwargs": kwargs,
                        },
                        "model_key": key,
                    })

        if not primary_tasks and not verifier_keys:
            logger.warning("No model tasks to execute!")
            return []

        primary_results = self._execute_model_tasks(primary_tasks) if primary_tasks else []

        # Handle fallbacks for failed primary models
        self._handle_fallbacks(primary_results, query)

        valid_primary = [r for r in primary_results if r.get("text", "").strip()]

        # ── Phase 2: Verification against actual output ─────────
        if verifier_keys and valid_primary:
            # Build the text that verifiers should review
            best_primary = max(
                valid_primary,
                key=lambda r: len(r.get("text", "")),
            )
            primary_output = best_primary.get("text", "")

            verifier_tasks = []
            for key, role in verifier_keys:
                engine = self._models.get(key)
                if engine is None:
                    continue

                if isinstance(engine, ClinicalCamelEngine):
                    fn = engine.validate
                    kwargs = {"query": query, "model_output": primary_output}
                elif isinstance(engine, Med42Engine):
                    fn = engine.cross_check
                    kwargs = {"query": query, "analysis": primary_output}
                else:
                    fn = engine.generate
                    kwargs = {"prompt": f"Verify this medical analysis:\n\n{primary_output}"}

                verifier_tasks.append({
                    "name": f"{key}::{role}",
                    "fn": self._managed_model_call,
                    "kwargs": {
                        "model_key": key,
                        "callable_fn": fn,
                        "call_kwargs": kwargs,
                    },
                    "model_key": key,
                })

            if verifier_tasks:
                verifier_results = self._execute_model_tasks(verifier_tasks)
                self._handle_fallbacks(verifier_results, query)
                primary_results.extend(verifier_results)

        # Filter out error-only results
        valid = [r for r in primary_results if r.get("text", "").strip()]
        return valid if valid else primary_results

    def _dispatch_engine(
        self,
        engine: BaseLLM,
        key: str,
        role: str,
        query: str,
        rag_context: str,
        route: dict[str, Any],
        conversation_history: list[dict[str, str]] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Determine the correct method and kwargs for an engine."""
        conversation_history = conversation_history or []

        if isinstance(engine, DeepSeekEngine) and route.get("enable_self_reflection"):
            return engine.reason, {"query": query, "rag_context": rag_context}
        elif isinstance(engine, MeditronEngine):
            return engine.clinical_analysis, {"query": query, "rag_context": rag_context}
        elif isinstance(engine, OpenBioLLMEngine):
            return engine.analyze, {"query": query, "rag_context": rag_context}
        elif isinstance(engine, PMCLLaMAEngine):
            return engine.synthesize_evidence, {"query": query, "evidence": rag_context}
        elif isinstance(engine, MeLLaMAEngine):
            return engine.answer, {
                "query": query,
                "rag_context": rag_context,
                "conversation_history": conversation_history,
            }
        elif isinstance(engine, BioMistralEngine):
            return engine.quick_answer, {"query": query}
        elif isinstance(engine, ChatDoctorEngine):
            return engine.chat, {
                "message": query,
                "conversation_history": conversation_history,
            }
        else:
            return engine.generate, {"prompt": query}

    def _is_heavy_model(self, model_key: str) -> bool:
        """Identify models that should not be loaded in parallel on shared GPUs."""
        engine = self._models.get(model_key)
        if engine is None:
            return False

        memory_tier = str(engine.config.get("memory_tier", "")).strip().lower()
        if memory_tier in {"heavy", "large", "xl", "giant"}:
            return True
        if memory_tier in {"light", "small", "medium"}:
            return False

        descriptor = f"{model_key} {engine.model_id}".lower()
        return any(token in descriptor for token in ("70b", "72b", "65b"))

    def _touch_resident_model(self, model_key: str) -> None:
        """Update LRU order for a loaded model."""
        self._resident_models.pop(model_key, None)
        self._resident_models[model_key] = None

    def _prepare_model_for_inference(self, model_key: str) -> None:
        """Load a model with LRU-based GPU residency control."""
        engine = self._models[model_key]

        with self._model_state_lock:
            self._active_model_calls[model_key] = self._active_model_calls.get(model_key, 0) + 1
            if engine.is_loaded:
                self._touch_resident_model(model_key)
                return

            try:
                if self._max_resident_models > 0:
                    loaded_models = [
                        key for key in self._resident_models
                        if self._models.get(key) is not None and self._models[key].is_loaded
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
                        logger.info("Evicting resident model before load: %s", evicted_key)
                        self._models[evicted_key].unload()
                        self._resident_models.pop(evicted_key, None)
                        loaded_models = [
                            key for key in self._resident_models
                            if self._models.get(key) is not None and self._models[key].is_loaded
                        ]

                logger.info("Preparing model for inference: %s", model_key)
                engine.load()
                self._touch_resident_model(model_key)
            except Exception:
                active = max(0, self._active_model_calls.get(model_key, 1) - 1)
                if active == 0:
                    self._active_model_calls.pop(model_key, None)
                else:
                    self._active_model_calls[model_key] = active
                raise

    def _release_model_after_inference(self, model_key: str) -> None:
        """Update active usage counters and optionally unload the model."""
        engine = self._models[model_key]

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
            if should_unload and active == 0 and engine.is_loaded:
                logger.info("Auto-unloading model after inference: %s", model_key)
                engine.unload()
                self._resident_models.pop(model_key, None)
            elif engine.is_loaded:
                self._touch_resident_model(model_key)

    def _managed_model_call(
        self,
        model_key: str,
        callable_fn: Any,
        call_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Wrap model calls so lazy loading also enforces GPU residency policy."""
        self._prepare_model_for_inference(model_key)
        try:
            return callable_fn(**call_kwargs)
        finally:
            self._release_model_after_inference(model_key)

    def _execute_model_tasks(self, tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Run light tasks in parallel and heavy tasks sequentially when configured."""
        if not tasks:
            return []

        if not self._sequential_heavy_models:
            return self._executor.execute(tasks)

        light_tasks = [task for task in tasks if not self._is_heavy_model(task.get("model_key", ""))]
        heavy_tasks = [task for task in tasks if self._is_heavy_model(task.get("model_key", ""))]

        results: list[dict[str, Any]] = []
        if light_tasks:
            results.extend(self._executor.execute(light_tasks))
        if heavy_tasks:
            logger.info("Executing %d heavy model task(s) sequentially", len(heavy_tasks))
            results.extend(self._executor.execute_sequential(heavy_tasks))
        return results

    def _handle_fallbacks(
        self, results: list[dict[str, Any]], query: str,
    ) -> None:
        """Process fallbacks for failed model results."""
        fallback_results = []
        for result in results:
            task_name = result.get("task_name", "")
            model_key = task_name.rsplit("::", 1)[0] if "::" in task_name else task_name

            if result.get("error"):
                self._fallback.record_failure(model_key)
                fallback_key = self._fallback.get_fallback(model_key)
                if fallback_key and fallback_key in self._models:
                    logger.info(f"Trying fallback: {fallback_key}")
                    try:
                        fb_engine = self._models[fallback_key]
                        fb_result = self._managed_model_call(
                            fallback_key,
                            fb_engine.generate,
                            {"prompt": query},
                        )
                        fb_result["task_name"] = f"{fallback_key}::fallback"
                        fallback_results.append(fb_result)
                    except Exception as e:
                        logger.error(f"Fallback {fallback_key} also failed: {e}")
            else:
                self._fallback.record_success(model_key)

        results.extend(fallback_results)

    def _layer4_fusion(
        self,
        model_results: list[dict[str, Any]],
        rag_evidence: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Layer 4: Fuse model outputs."""
        fused = self._fusion.fuse(model_results, rag_evidence)

        # Add uncertainty estimation
        uncertainty = self._uncertainty.estimate(model_results)
        fused["uncertainty_detail"] = uncertainty

        # Add contradiction detection
        contradictions = self._contradictions.detect(model_results)
        fused["contradictions"] = contradictions

        return fused

    def _layer5_safety(
        self,
        query: str,
        fused: dict[str, Any],
        input_analysis: dict[str, Any],
        rag_evidence: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Layer 5: Safety validation."""
        consensus_text = fused.get("consensus_answer", fused.get("text", ""))

        # Clinical validation
        validation = self._clinical_validator.validate(consensus_text)

        # Risk flagging
        risk = self._risk_flagger.flag(consensus_text)

        # Drug interaction check
        medications = input_analysis.get("medications", [])
        drug_names = [m.get("drug", "") for m in medications if m.get("drug")]
        drug_check = self._drug_checker.check_interactions(drug_names) if drug_names else {}

        # Hallucination detection (if RAG evidence available)
        hallucination = {}
        if rag_evidence:
            hallucination = self._hallucination.verify(consensus_text, query)
        else:
            validation.setdefault("warnings", []).append(
                "No grounded evidence retrieved — response relies on model priors and local context.",
            )

        return {
            **validation,
            **risk,
            "drug_check": drug_check,
            "hallucination_check": hallucination,
        }

    def _layer6_response(
        self,
        fused: dict[str, Any],
        safety_result: dict[str, Any],
        rag_evidence: list[dict[str, Any]],
        mode: str,
        retrieval_sources: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Layer 6: Generate structured report."""
        report = self._report_generator.generate(
            fused_result=fused,
            safety_result=safety_result,
            rag_evidence=rag_evidence,
            drug_check=safety_result.get("drug_check"),
            mode=mode,
            sources=retrieval_sources,
        )

        # UX-1 FIX: Apply ResponseStyler for audience-appropriate formatting
        if report.get("report_text"):
            report["report_text"] = self._response_styler.style(
                report["report_text"], mode=mode
            )

        return report

    def _layer7_audit(
        self,
        query: str,
        input_analysis: dict[str, Any],
        route: dict[str, Any],
        model_results: list[dict[str, Any]],
        fused: dict[str, Any],
        safety_result: dict[str, Any],
        report: dict[str, Any],
        execution_time: float,
    ) -> str:
        """Layer 7: Audit logging."""
        return self._audit.log_analysis(
            query=query,
            query_category=input_analysis.get("query_category", "unknown"),
            routing_decision=route,
            model_results=model_results,
            fused_result=fused,
            safety_result=safety_result,
            report=report,
            execution_time=execution_time,
        )

    # ═══════════════════════════════════════════════════════════
    #  UTILITIES
    # ═══════════════════════════════════════════════════════════

    def ingest_knowledge(self, directory: str) -> int:
        """Ingest documents into the RAG knowledge base."""
        if self._rag:
            self._rag.initialize()
            return self._rag.ingest_directory(directory)
        return 0

    def search_knowledge_base(
        self,
        query: str,
        max_results: int = 10,
        sources: list[str] | None = None,
    ) -> dict[str, Any]:
        """Expose the advanced retrieval pipeline for platform search routes."""
        if self._retrieval_pipeline is None:
            return {"sources": [], "total": 0, "query_analysis": {}, "warnings": []}

        retrieval = self._retrieval_pipeline.retrieve(
            query,
            max_results=max_results,
            requested_sources=sources,
        )
        payload = retrieval.to_dict()
        payload["total"] = len(payload.get("sources", []))
        return payload

    def query_vector_db(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Query the configured vector store directly."""
        if not self._rag:
            return []

        self._rag.initialize()
        results = self._rag.query(query, top_k=top_k)
        return [
            {
                "title": item.get("metadata", {}).get("title", "Knowledge Base"),
                "content": item.get("content", ""),
                "score": item.get("relevance", 0.0),
                "source": item.get("metadata", {}).get("source", "vector_db"),
            }
            for item in results
        ]

    def ingest_built_in_knowledge(self) -> int:
        """Ingest built-in clinical guidelines."""
        if self._rag and self._knowledge_base:
            self._rag.initialize()
            return self._knowledge_base.ingest_built_in(self._rag)
        return 0

    def get_model_status(self) -> dict[str, Any]:
        """Get health status of all registered models."""
        return {
            key: engine.health_check()
            for key, engine in self._models.items()
        }

    def unload_model(self, model_key: str) -> None:
        """Unload a specific model from memory."""
        if model_key in self._models:
            self._models[model_key].unload()
            with self._model_state_lock:
                self._resident_models.pop(model_key, None)
                self._active_model_calls.pop(model_key, None)

    def unload_all(self) -> None:
        """Unload all models."""
        for engine in self._models.values():
            engine.unload()
        with self._model_state_lock:
            self._resident_models.clear()
            self._active_model_calls.clear()


# ═══════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def main():
    """Command-line interface for the Medical LLM Engine."""
    parser = argparse.ArgumentParser(
        description="🏥 Medical LLM Super-Engine — Text-to-Text Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --query "Patient presents with chest pain radiating to left arm"
  python main.py --query "What are the side effects of metformin?" --mode patient
  python main.py --interactive
  python main.py --ingest ./data/guidelines
        """,
    )
    parser.add_argument(
        "--query", "-q", type=str,
        help="Medical query to analyze",
    )
    parser.add_argument(
        "--mode", "-m", type=str, default="doctor",
        choices=["doctor", "patient", "research"],
        help="Output audience mode (default: doctor)",
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true",
        help="Run in interactive mode",
    )
    parser.add_argument(
        "--ingest", type=str,
        help="Directory to ingest into RAG knowledge base",
    )
    parser.add_argument(
        "--model-config", type=str, default="config/model_config.yaml",
        help="Path to model config file",
    )
    parser.add_argument(
        "--pipeline-config", type=str, default="config/pipeline_config.yaml",
        help="Path to pipeline config file",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--no-rag", action="store_true",
        help="Disable RAG retrieval",
    )
    parser.add_argument(
        "--models", type=str, nargs="*",
        help="Force specific models (space-separated keys)",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s │ %(name)-20s │ %(levelname)-7s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Initialize engine
    engine = MedicalLLMEngine(
        model_config_path=args.model_config,
        pipeline_config_path=args.pipeline_config,
    )
    engine.initialize()

    # Ingest knowledge if requested
    if args.ingest:
        count = engine.ingest_knowledge(args.ingest)
        logger.info(f"Ingested {count} chunks from {args.ingest}")

    # Ingest built-in knowledge
    engine.ingest_built_in_knowledge()

    if args.interactive:
        _interactive_mode(engine, args.mode)
    elif args.query:
        result = engine.analyze(
            query=args.query,
            mode=args.mode,
            enable_rag=not args.no_rag,
            force_models=args.models,
        )
        print(result.get("report_text", "No report generated."))
    else:
        parser.print_help()


def _interactive_mode(engine: MedicalLLMEngine, default_mode: str) -> None:
    """Interactive CLI mode."""
    print("\n" + "═" * 60)
    print("  🏥 Medical LLM Super-Engine — Interactive Mode")
    print("  Type 'quit' to exit, 'mode <doctor|patient|research>' to change mode")
    print("═" * 60)

    mode = default_mode

    while True:
        try:
            query = input(f"\n[{mode}] 🩺 Enter query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("👋 Goodbye!")
            break
        if query.lower().startswith("mode "):
            new_mode = query.split(maxsplit=1)[1].strip()
            if new_mode in ("doctor", "patient", "research"):
                mode = new_mode
                print(f"✅ Mode changed to: {mode}")
            else:
                print("❌ Invalid mode. Use: doctor, patient, research")
            continue

        result = engine.analyze(
            query=query,
            mode=mode,
            session_id="interactive_cli",
        )
        print(result.get("report_text", "No report generated."))


if __name__ == "__main__":
    main()
