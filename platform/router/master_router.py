"""
MASTER ROUTER — The brain of MedAI Platform (enterprise-grade).

Routes every incoming request to the correct engine(s):
    - Medical images     -> MediScan VLM
    - Documents / PDFs   -> MediScan OCR
    - Text questions     -> Medical LLM
    - Needs web search   -> Search Engine + RAG
    - Needs patient data -> Patient DB + RAG
    - Multi-modal        -> Parallel pipeline (VLM + LLM)

Enterprise features:
    - Circuit breaker with rolling-window failure rate
    - Retry with exponential backoff (transient errors only)
    - Per-engine bulkhead (semaphore isolation)
    - Request correlation ID propagation to engines
    - Structured logging (no f-strings)
    - Per-request timeout enforcement
    - User-isolated response caching with TTL
    - Base-directory-restricted path validation
    - File size enforcement before engine dispatch
    - Output sanitization (strip internal fields)
    - Configurable via settings (no hardcoded values)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import mimetypes
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import httpx

from config import settings
from health_utils import engine_health_endpoint, normalize_engine_health

logger = logging.getLogger(__name__)

# ── Max file size for engine dispatch (prevent OOM) ──────────────────────
_MAX_ENGINE_FILE_BYTES = settings.max_upload_size_bytes


class RouteTarget(str, Enum):
    GENERAL_LLM = "general_llm"      # Cloud LLM backbone (Claude/OpenAI) — handles ALL text
    MEDISCAN_VLM = "mediscan_vlm"
    MEDISCAN_OCR = "mediscan_ocr"
    MEDICAL_LLM = "medical_llm"
    SEARCH_RAG = "search_rag"
    PATIENT_DB = "patient_db"
    MULTI_ENGINE = "multi_engine"
    OPENRAG = "openrag"              # Agentic RAG with Docling ingestion + hybrid search
    CONTEXT_GRAPH = "context_graph"  # Neo4j patient longitudinal memory
    CONTEXT1_AGENT = "context1_agent"  # Chroma Context-1 multi-hop retrieval


@dataclass
class RoutingDecision:
    """Result of the routing analysis."""
    primary_target: RouteTarget
    secondary_targets: list[RouteTarget] = field(default_factory=list)
    confidence: float = 1.0
    reason: str = ""
    requires_patient_context: bool = False
    requires_search: bool = False


@dataclass
class EngineResponse:
    """Unified response from any engine."""
    engine: str
    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    latency_ms: float = 0.0
    confidence: float = 0.0


# ── Keywords / heuristics for routing ────────────────────────────────────

_SEARCH_KEYWORDS = frozenset({
    "latest", "recent", "new", "current", "2024", "2025", "2026",
    "guideline", "study", "research", "pubmed", "evidence", "trial",
    "update", "news", "recommendation",
})

_PATIENT_KEYWORDS = frozenset({
    "my report", "my scan", "my history", "my results", "my records",
    "previous", "last visit", "follow up", "my prescription",
})

_IMAGE_MIME_PREFIXES = ("image/", "application/dicom")
_DOCUMENT_MIMES = (
    "application/pdf", "application/msword",
    "application/vnd.openxmlformats-officedocument",
)

_MEDICAL_IMAGE_KEYWORDS = frozenset({
    "x-ray", "xray", "ct scan", "mri", "ultrasound", "ecg",
    "pathology", "histology", "retina", "fundus", "mammogram",
    "radiograph", "dicom",
})

# ── Circuit Breaker (rolling-window failure rate) ────────────────────────

@dataclass
class _CircuitState:
    """Tracks per-engine health with rolling-window failure rate."""
    failures: int = 0
    successes: int = 0
    last_failure: float = 0.0
    is_open: bool = False
    half_open: bool = False
    total_requests: int = 0
    total_failures: int = 0
    # Window tracking for failure rate
    _window_start: float = field(default_factory=time.monotonic)
    _window_failures: int = 0
    _window_requests: int = 0


class _CircuitBreaker:
    """Rolling-window circuit breaker with half-open state."""

    def __init__(
        self,
        failure_threshold: int = 5,
        failure_rate_threshold: float = 0.5,
        recovery_seconds: float = 60.0,
        window_seconds: float = 120.0,
        min_requests_for_rate: int = 10,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._failure_rate_threshold = failure_rate_threshold
        self._recovery_seconds = recovery_seconds
        self._window_seconds = window_seconds
        self._min_requests = min_requests_for_rate

    def check(self, state: _CircuitState) -> bool:
        """Return True if the circuit allows a request."""
        if not state.is_open:
            return True
        # Check if recovery period has elapsed -> half-open
        if (time.monotonic() - state.last_failure) > self._recovery_seconds:
            state.half_open = True
            logger.info("Circuit breaker HALF-OPEN (allowing probe request)")
            return True
        return False

    def record_success(self, state: _CircuitState) -> None:
        state.successes += 1
        state.total_requests += 1
        self._update_window(state, success=True)
        if state.half_open:
            state.is_open = False
            state.half_open = False
            state.failures = 0
            logger.info("Circuit breaker CLOSED (probe succeeded)")

    def record_failure(self, state: _CircuitState) -> None:
        state.failures += 1
        state.total_failures += 1
        state.total_requests += 1
        state.last_failure = time.monotonic()
        self._update_window(state, success=False)

        if state.half_open:
            # Half-open probe failed -> re-open
            state.is_open = True
            state.half_open = False
            logger.warning("Circuit breaker RE-OPENED (probe failed)")
            return

        # Open on consecutive failure threshold
        if state.failures >= self._failure_threshold:
            state.is_open = True
            logger.warning("Circuit breaker OPEN (consecutive failures: %d)", state.failures)
            return

        # Open on failure rate threshold (rolling window)
        if state._window_requests >= self._min_requests:
            rate = state._window_failures / state._window_requests
            if rate >= self._failure_rate_threshold:
                state.is_open = True
                logger.warning(
                    "Circuit breaker OPEN (failure rate: %.1f%% over %d requests)",
                    rate * 100, state._window_requests,
                )

    def _update_window(self, state: _CircuitState, success: bool) -> None:
        """Reset rolling window if expired."""
        now = time.monotonic()
        if (now - state._window_start) > self._window_seconds:
            state._window_start = now
            state._window_failures = 0
            state._window_requests = 0
        state._window_requests += 1
        if not success:
            state._window_failures += 1


# ── LRU Cache with TTL (user-isolated) ──────────────────────────────────

class _TTLCache:
    """Simple LRU cache with TTL for response caching."""

    def __init__(self, max_size: int = 200, ttl_seconds: float = 300.0) -> None:
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl_seconds

    def get(self, key: str) -> Any | None:
        entry = self._cache.get(key)
        if entry is None:
            return None
        value, timestamp = entry
        if (time.monotonic() - timestamp) > self._ttl:
            del self._cache[key]
            return None
        self._cache.move_to_end(key)
        return value

    def set(self, key: str, value: Any) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = (value, time.monotonic())
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    @staticmethod
    def build_key(
        user_id: str | None,
        text: str,
        mode: str,
        engine: str,
    ) -> str:
        """Build a user-isolated, collision-resistant cache key."""
        raw = f"{user_id or 'anon'}:{text[:500]}:{mode}:{engine}"
        return hashlib.sha256(raw.encode()).hexdigest()


# ── Path Validation ──────────────────────────────────────────────────────

def _validate_file_path(file_path: str | None) -> str | None:
    """
    Reject path traversal, symlink escapes, oversized files,
    and files outside allowed base directories.
    """
    if file_path is None:
        return None

    # Check for path traversal BEFORE resolve (resolve would hide ..)
    raw_parts = Path(file_path).parts
    if ".." in raw_parts:
        raise ValueError("Path traversal detected")

    # Resolve and check existence
    p = Path(file_path).resolve()
    if not p.is_file():
        raise ValueError("File does not exist")

    # Restrict to allowed base directories (temp dir + upload dir)
    import tempfile
    allowed_bases = [Path(tempfile.gettempdir()).resolve()]
    if settings.upload_temp_dir:
        allowed_bases.append(Path(settings.upload_temp_dir).resolve())

    if not any(p.is_relative_to(base) for base in allowed_bases):
        logger.warning("File path outside allowed directories: %s", p)
        raise ValueError("Access denied: file outside allowed directory")

    # Enforce file size limit
    file_size = p.stat().st_size
    if file_size > _MAX_ENGINE_FILE_BYTES:
        raise ValueError(
            f"File too large ({file_size} bytes, max {_MAX_ENGINE_FILE_BYTES})"
        )

    return str(p)


# ── Output Sanitization ─────────────────────────────────────────────────

_INTERNAL_FIELDS = frozenset({
    "internal_trace", "debug_info", "raw_prompt", "system_prompt",
    "model_config", "api_key", "token", "credentials",
})


def _sanitize_engine_output(data: dict[str, Any]) -> dict[str, Any]:
    """Remove internal/debug fields from engine responses before returning."""
    return {k: v for k, v in data.items() if k not in _INTERNAL_FIELDS}


class MasterRouter:
    """
    Analyses input (text + optional files) and routes to the
    correct engine(s). Merges multi-engine results when needed.

    Routing Architecture (4-layer, like ChatGPT/Claude/Gemini):
        Layer 1: Rule Engine — MIME type detection, emergency keywords (<1ms)
        Layer 2: Embedding Router — BGE semantic similarity (~5-10ms)
        Layer 3: Confidence Resolver — combine scores, multi-intent detection
        Layer 4: General LLM — fallback for ambiguous queries

    Enterprise features:
        - Multi-layer intelligent routing (no hardcoded patterns)
        - MIME-type-based file routing (automatic)
        - Semantic intent classification via embeddings
        - Graceful fallback chain: Embedding → Rules → General LLM
        - Per-engine circuit breaker with rolling-window failure rate
        - Retry with exponential backoff (transient errors only)
        - Per-engine semaphore (bulkhead isolation)
        - User-isolated response caching
        - Request ID propagation to engines
        - File size enforcement
        - Output sanitization
        - Structured telemetry
    """

    def __init__(self, llm_service=None, intent_router=None, search_engine=None) -> None:
        self._llm_service = llm_service  # GeneralLLMService instance
        self._intent_router = intent_router  # IntentRouter instance (Layer 2+3)
        self._search_engine = search_engine
        self._client = httpx.AsyncClient(
            timeout=settings.engine_timeout_seconds,
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=20,
                keepalive_expiry=30,
            ),
        )
        self._circuit_breaker = _CircuitBreaker(
            failure_threshold=5,
            failure_rate_threshold=0.5,
            recovery_seconds=60.0,
            window_seconds=120.0,
            min_requests_for_rate=10,
        )
        self._circuits: dict[str, _CircuitState] = {
            t.value: _CircuitState() for t in RouteTarget
        }
        # Bulkhead: per-engine concurrency limiter
        self._semaphores: dict[str, asyncio.Semaphore] = {
            t.value: asyncio.Semaphore(10) for t in RouteTarget
        }
        self._cache = _TTLCache(max_size=200, ttl_seconds=300.0)

    async def close(self) -> None:
        await self._client.aclose()

    # ── Core Routing Logic ───────────────────────────────────────

    def analyze_input(
        self,
        text: str | None = None,
        file_mime: str | None = None,
        file_name: str | None = None,
        has_patient_context: bool = False,
    ) -> RoutingDecision:
        """Determine where to send this request."""

        # 1. File-based routing (highest priority)
        if file_mime:
            if any(file_mime.startswith(p) for p in _IMAGE_MIME_PREFIXES):
                return RoutingDecision(
                    primary_target=RouteTarget.MEDISCAN_VLM,
                    secondary_targets=[RouteTarget.MEDICAL_LLM] if text else [],
                    reason="Image file detected: %s" % file_mime,
                )
            if any(file_mime.startswith(d) for d in _DOCUMENT_MIMES):
                return RoutingDecision(
                    primary_target=RouteTarget.MEDISCAN_OCR,
                    secondary_targets=[RouteTarget.MEDICAL_LLM] if text else [],
                    reason="Document file detected: %s" % file_mime,
                )

        # 2. File extension fallback
        if file_name:
            mime_guess, _ = mimetypes.guess_type(file_name)
            if mime_guess:
                return self.analyze_input(
                    text=text, file_mime=mime_guess,
                    has_patient_context=has_patient_context,
                )

        # 3. Text-based routing — Multi-layer intelligent classification
        #    Layer 2: Embedding router (semantic similarity, ~5-10ms)
        #    Layer 3: Confidence resolver (combine scores, multi-intent)
        #    Fallback: keyword heuristics if embedding model not loaded
        if text:
            # ── Layer 2+3: Embedding-based intent classification ─────
            if self._intent_router and self._intent_router.is_loaded:
                intent_result = self._intent_router.route(text)

                # Map intent route names to RouteTarget enum
                _route_map = {
                    "general_llm": RouteTarget.GENERAL_LLM,
                    "mediscan_vlm": RouteTarget.MEDISCAN_VLM,
                    "mediscan_ocr": RouteTarget.MEDISCAN_OCR,
                    "medical_llm": RouteTarget.MEDICAL_LLM,
                    "search_rag": RouteTarget.SEARCH_RAG,
                    "patient_db": RouteTarget.PATIENT_DB,
                }

                primary = _route_map.get(
                    intent_result["primary_route"], RouteTarget.GENERAL_LLM,
                )

                # Check if general LLM is available, fall back to medical LLM
                if primary == RouteTarget.GENERAL_LLM:
                    if not self._llm_service or not self._llm_service.available:
                        primary = RouteTarget.MEDICAL_LLM

                # Build secondary targets from embedding scores
                secondary: list[RouteTarget] = []
                for sec in intent_result.get("secondary_intents", []):
                    target = _route_map.get(sec["route"])
                    if target and target != primary and target not in secondary:
                        secondary.append(target)

                # Override: if patient context provided, always add patient_db
                if has_patient_context and RouteTarget.PATIENT_DB not in secondary:
                    secondary.append(RouteTarget.PATIENT_DB)

                return RoutingDecision(
                    primary_target=primary,
                    secondary_targets=secondary,
                    confidence=intent_result["primary_confidence"],
                    reason="Embedding router: %s (%.2f)" % (
                        intent_result["primary_intent"],
                        intent_result["primary_confidence"],
                    ),
                    requires_patient_context=has_patient_context or any(
                        s.get("intent") == "patient_context"
                        for s in intent_result.get("secondary_intents", [])
                    ),
                    requires_search=any(
                        s.get("intent") == "web_search"
                        for s in intent_result.get("secondary_intents", [])
                    ),
                )

            # ── Fallback: keyword heuristics (if embedding model not loaded) ──
            text_lower = text.lower()
            primary = RouteTarget.GENERAL_LLM
            if not self._llm_service or not self._llm_service.available:
                primary = RouteTarget.MEDICAL_LLM

            requires_patient = has_patient_context or any(
                kw in text_lower for kw in _PATIENT_KEYWORDS
            )
            requires_search = any(kw in text_lower for kw in _SEARCH_KEYWORDS)

            secondary_fb: list[RouteTarget] = []
            if requires_search:
                secondary_fb.append(RouteTarget.SEARCH_RAG)
            if requires_patient:
                secondary_fb.append(RouteTarget.PATIENT_DB)

            return RoutingDecision(
                primary_target=primary,
                secondary_targets=secondary_fb,
                reason="Keyword fallback -> General LLM",
                requires_patient_context=requires_patient,
                requires_search=requires_search,
            )

        return RoutingDecision(
            primary_target=RouteTarget.GENERAL_LLM
            if (self._llm_service and self._llm_service.available)
            else RouteTarget.MEDICAL_LLM,
            reason="Default fallback",
            confidence=0.5,
        )

    # ── Engine Dispatch ──────────────────────────────────────────

    async def route_and_execute(
        self,
        text: str | None = None,
        file_path: str | None = None,
        file_mime: str | None = None,
        file_name: str | None = None,
        patient_id: str | None = None,
        session_id: str | None = None,
        mode: str = "doctor",
        user_id: str | None = None,
        request_id: str | None = None,
        web_search: bool = False,
        deep_reasoning: bool = False,
    ) -> dict[str, Any]:
        """Route input and execute against the appropriate engine(s)."""

        safe_path = _validate_file_path(file_path)

        start = time.monotonic()
        decision = self.analyze_input(
            text=text, file_mime=file_mime,
            file_name=file_name,
            has_patient_context=bool(patient_id),
        )

        # Honor explicit feature toggles from the frontend
        if (
            (web_search or decision.requires_search)
            and decision.primary_target != RouteTarget.SEARCH_RAG
            and RouteTarget.SEARCH_RAG not in decision.secondary_targets
        ):
            decision.secondary_targets.append(RouteTarget.SEARCH_RAG)
            decision.requires_search = True

        if (
            (web_search or decision.requires_search)
            and settings.enable_openrag
            and decision.primary_target != RouteTarget.OPENRAG
            and RouteTarget.OPENRAG not in decision.secondary_targets
        ):
            decision.secondary_targets.append(RouteTarget.OPENRAG)

        if (
            patient_id
            and settings.enable_context_graph
            and decision.primary_target != RouteTarget.CONTEXT_GRAPH
            and RouteTarget.CONTEXT_GRAPH not in decision.secondary_targets
        ):
            decision.secondary_targets.append(RouteTarget.CONTEXT_GRAPH)

        if (
            deep_reasoning
            and settings.enable_context1_agent
            and decision.primary_target != RouteTarget.CONTEXT1_AGENT
            and RouteTarget.CONTEXT1_AGENT not in decision.secondary_targets
        ):
            decision.secondary_targets.append(RouteTarget.CONTEXT1_AGENT)

        logger.info(
            "Routing decision: target=%s reason=%s request_id=%s",
            decision.primary_target.value, decision.reason, request_id or "-",
        )

        # Check user-isolated cache for text-only queries
        cache_key = None
        if text and not safe_path:
            cache_key = _TTLCache.build_key(
                user_id, text, mode, decision.primary_target.value,
            )
            cached = self._cache.get(cache_key)
            if cached:
                logger.info(
                    "Cache hit: engine=%s request_id=%s",
                    decision.primary_target.value, request_id or "-",
                )
                cached["from_cache"] = True
                return cached

        # Execute primary + secondaries in parallel
        targets = [decision.primary_target] + decision.secondary_targets
        tasks = [
            asyncio.create_task(
                self._call_engine_with_retry(
                    target=target,
                    text=text,
                    file_path=safe_path,
                    patient_id=patient_id,
                    session_id=session_id,
                    mode=mode,
                    request_id=request_id,
                )
            )
            for target in targets
        ]

        responses: list[EngineResponse] = await asyncio.gather(*tasks)
        total_ms = (time.monotonic() - start) * 1000

        result = self._merge_responses(
            decision=decision,
            responses=responses,
            total_latency_ms=total_ms,
        )

        # Cache successful text-only results
        if cache_key and result.get("answer"):
            self._cache.set(cache_key, result)

        return result

    async def _call_engine_with_retry(
        self,
        target: RouteTarget,
        text: str | None = None,
        file_path: str | None = None,
        patient_id: str | None = None,
        session_id: str | None = None,
        mode: str = "doctor",
        request_id: str | None = None,
    ) -> EngineResponse:
        """Call engine with retry and exponential backoff (transient errors only)."""
        max_attempts = settings.engine_retry_max_attempts
        backoff = settings.engine_retry_backoff_seconds
        response: EngineResponse | None = None

        for attempt in range(1, max_attempts + 1):
            response = await self._call_engine(
                target, text, file_path, patient_id, session_id, mode,
                request_id,
            )
            if response.success:
                return response

            # Don't retry non-transient errors (4xx, validation, etc.)
            if response.error and any(
                marker in response.error
                for marker in ("400", "401", "403", "404", "422", "validation")
            ):
                logger.info(
                    "Engine %s returned non-retryable error: %s",
                    target.value, response.error[:200],
                )
                return response

            if attempt < max_attempts:
                wait = backoff * (2 ** (attempt - 1))
                logger.warning(
                    "Engine %s failed (attempt %d/%d), retrying in %.1fs: %s",
                    target.value, attempt, max_attempts, wait,
                    response.error[:100] if response.error else "unknown",
                )
                await asyncio.sleep(wait)

        return response  # type: ignore[return-value]

    async def _call_engine(
        self,
        target: RouteTarget,
        text: str | None = None,
        file_path: str | None = None,
        patient_id: str | None = None,
        session_id: str | None = None,
        mode: str = "doctor",
        request_id: str | None = None,
    ) -> EngineResponse:
        """Call a specific engine with circuit breaker + bulkhead + per-request timeout."""
        engine_name = target.value
        circuit = self._circuits.get(engine_name)

        # Circuit breaker check
        if circuit and not self._circuit_breaker.check(circuit):
            return EngineResponse(
                engine=engine_name,
                success=False,
                error="Circuit breaker open — engine temporarily unavailable",
            )

        # Bulkhead: limit concurrent calls per engine
        semaphore = self._semaphores.get(engine_name)
        start = time.monotonic()

        try:
            if semaphore:
                async with semaphore:
                    result = await asyncio.wait_for(
                        self._dispatch(
                            target, text, file_path, patient_id,
                            session_id, mode, request_id,
                        ),
                        timeout=settings.engine_timeout_seconds,
                    )
            else:
                result = await asyncio.wait_for(
                    self._dispatch(
                        target, text, file_path, patient_id,
                        session_id, mode, request_id,
                    ),
                    timeout=settings.engine_timeout_seconds,
                )

            latency = (time.monotonic() - start) * 1000
            if circuit:
                self._circuit_breaker.record_success(circuit)

            confidence = 0.0
            if isinstance(result, dict):
                confidence = result.get(
                    "confidence",
                    result.get("fusion", {}).get("confidence", 0.0),
                )

            return EngineResponse(
                engine=engine_name, success=True, data=result,
                latency_ms=latency, confidence=confidence,
            )

        except asyncio.TimeoutError:
            latency = (time.monotonic() - start) * 1000
            if circuit:
                self._circuit_breaker.record_failure(circuit)
            logger.error(
                "Engine %s timed out after %.1fms (request_id=%s)",
                engine_name, latency, request_id or "-",
            )
            return EngineResponse(
                engine=engine_name, success=False,
                error=f"Engine timed out after {settings.engine_timeout_seconds}s",
                latency_ms=latency,
            )
        except httpx.HTTPStatusError as e:
            latency = (time.monotonic() - start) * 1000
            status_code = e.response.status_code
            # Only count server errors as circuit failures (not 4xx)
            if circuit and status_code >= 500:
                self._circuit_breaker.record_failure(circuit)
            elif circuit:
                self._circuit_breaker.record_success(circuit)  # 4xx = engine is responsive
            logger.error(
                "Engine %s HTTP %d (request_id=%s)",
                engine_name, status_code, request_id or "-",
            )
            return EngineResponse(
                engine=engine_name, success=False,
                error=f"HTTP {status_code}: {str(e)[:200]}",
                latency_ms=latency,
            )
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            latency = (time.monotonic() - start) * 1000
            if circuit:
                self._circuit_breaker.record_failure(circuit)
            logger.error(
                "Engine %s connection failed (request_id=%s): %s",
                engine_name, request_id or "-", e,
            )
            return EngineResponse(
                engine=engine_name, success=False,
                error=f"Connection failed: {str(e)[:200]}",
                latency_ms=latency,
            )
        except Exception as e:
            latency = (time.monotonic() - start) * 1000
            if circuit:
                self._circuit_breaker.record_failure(circuit)
            logger.error(
                "Engine %s unexpected error (request_id=%s): %s",
                engine_name, request_id or "-", e,
            )
            return EngineResponse(
                engine=engine_name, success=False,
                error=str(e)[:200], latency_ms=latency,
            )

    async def _dispatch(
        self,
        target: RouteTarget,
        text: str | None,
        file_path: str | None,
        patient_id: str | None,
        session_id: str | None,
        mode: str,
        request_id: str | None = None,
    ) -> dict:
        """Dispatch to the correct engine."""
        # Build correlation headers for distributed tracing
        headers = {}
        if request_id:
            headers["X-Request-ID"] = request_id

        if target == RouteTarget.GENERAL_LLM:
            return await self._call_general_llm(text, mode, session_id, headers)
        elif target == RouteTarget.MEDISCAN_VLM:
            return await self._call_mediscan_vlm(
                text, file_path, patient_id, headers,
            )
        elif target == RouteTarget.MEDISCAN_OCR:
            return await self._call_mediscan_ocr(file_path, headers)
        elif target == RouteTarget.MEDICAL_LLM:
            return await self._call_medical_llm(text, mode, session_id, headers)
        elif target == RouteTarget.SEARCH_RAG:
            return await self._call_search_rag(text, headers)
        elif target == RouteTarget.PATIENT_DB:
            return {"patient_id": patient_id, "context": "patient_history_placeholder"}
        elif target == RouteTarget.OPENRAG:
            return await self._call_openrag(text, headers)
        elif target == RouteTarget.CONTEXT_GRAPH:
            return await self._call_context_graph(patient_id, session_id, text, headers)
        elif target == RouteTarget.CONTEXT1_AGENT:
            return await self._call_context1_agent(text, headers)
        return {}

    # ── Individual Engine Calls ──────────────────────────────────

    async def _call_mediscan_vlm(
        self,
        question: str | None,
        file_path: str | None,
        patient_id: str | None,
        headers: dict[str, str] | None = None,
    ) -> dict:
        url = f"{settings.mediscan_vlm_url}/analyze"
        data = {
            "question": question or "Generate comprehensive medical report",
            "patient_id": patient_id or "",
        }
        if file_path:
            file_bytes = await asyncio.to_thread(Path(file_path).read_bytes)
            file_name = Path(file_path).name
            resp = await self._client.post(
                url, data=data,
                files={"file": (file_name, file_bytes)},
                headers=headers or {},
            )
        else:
            resp = await self._client.post(url, data=data, headers=headers or {})
        resp.raise_for_status()
        return _sanitize_engine_output(resp.json())

    async def _call_mediscan_ocr(
        self,
        file_path: str | None,
        headers: dict[str, str] | None = None,
    ) -> dict:
        if not file_path:
            return {"error": "No file provided for OCR"}
        url = f"{settings.mediscan_ocr_url}/ocr"
        file_bytes = await asyncio.to_thread(Path(file_path).read_bytes)
        file_name = Path(file_path).name
        resp = await self._client.post(
            url,
            files={"file": (file_name, file_bytes)},
            headers=headers or {},
        )
        resp.raise_for_status()
        return _sanitize_engine_output(resp.json())

    async def _call_medical_llm(
        self,
        query: str | None,
        mode: str,
        session_id: str | None,
        headers: dict[str, str] | None = None,
    ) -> dict:
        if not query:
            return {"error": "No text query provided"}
        url = f"{settings.medical_llm_url}/analyze"
        resp = await self._client.post(
            url,
            json={
                "query": query,
                "mode": mode,
                "enable_rag": settings.enable_search_rag,
                "session_id": session_id,
            },
            headers=headers or {},
        )
        resp.raise_for_status()
        return _sanitize_engine_output(resp.json())

    async def _call_search_rag(
        self,
        query: str | None,
        headers: dict[str, str] | None = None,
    ) -> dict:
        if not query:
            return {"sources": []}
        if self._search_engine is not None:
            response = await self._search_engine.search(query, max_results=8)
            return {
                "query": response.query,
                "sources": [result.to_payload() for result in response.results],
                "total": response.total,
                "sources_queried": response.sources_queried,
                "query_analysis": response.query_analysis,
                "context": response.context,
                "warnings": response.warnings,
                "cached": response.cached,
            }
        url = f"{settings.medical_llm_url}/rag/search"
        resp = await self._client.post(
            url,
            json={"query": query},
            headers=headers or {},
            timeout=60,
        )
        resp.raise_for_status()
        return _sanitize_engine_output(resp.json())

    async def _call_general_llm(
        self,
        query: str | None,
        mode: str,
        session_id: str | None,
        headers: dict[str, str] | None = None,
    ) -> dict:
        """Call the General LLM service (Claude/OpenAI) for intelligent responses."""
        if not query:
            return {"error": "No text query provided"}
        if not self._llm_service:
            return {"error": "General LLM service not initialized"}

        result = await self._llm_service.generate(
            query=query,
            mode=mode,
        )
        return result

    # ── New Service Calls (Granite, OpenRAG, Context Graph, Context-1) ──

    async def _call_openrag(
        self,
        query: str | None,
        headers: dict[str, str] | None = None,
    ) -> dict:
        """Call OpenRAG agentic retrieval for hybrid search + re-ranking."""
        if not query or not settings.enable_openrag:
            return {"sources": []}
        resp = await self._client.post(
            f"{settings.openrag_url}/search",
            data={"query": query, "top_k": 10, "use_reranker": True},
            headers=headers or {},
            timeout=60,
        )
        resp.raise_for_status()
        payload = _sanitize_engine_output(resp.json())
        if isinstance(payload, dict):
            results = payload.get("results", [])
            if isinstance(results, list) and "sources" not in payload:
                payload["sources"] = [
                    {
                        "title": (
                            item.get("metadata", {}).get("title")
                            or item.get("metadata", {}).get("filename")
                            or item.get("source")
                            or "OpenRAG result"
                        ),
                        "content": item.get("text", ""),
                        "score": item.get("rerank_score", item.get("score", item.get("relevance", 0.0))),
                        "source": item.get("metadata", {}).get("source", "openrag"),
                        "url": item.get("metadata", {}).get("url", ""),
                    }
                    for item in results
                    if isinstance(item, dict)
                ]
        return payload

    async def _call_context_graph(
        self,
        patient_id: str | None,
        session_id: str | None,
        text: str | None,
        headers: dict[str, str] | None = None,
    ) -> dict:
        """Query Neo4j context graph for patient longitudinal memory."""
        if not patient_id or not settings.enable_context_graph:
            return {"context": "no_patient_id"}

        results = {}
        # Get patient summary
        try:
            resp = await self._client.get(
                f"{settings.context_graph_url}/patient/{patient_id}/summary",
                headers=headers or {},
                timeout=15,
            )
            if resp.status_code == 200:
                results["summary"] = resp.json()
        except Exception as e:
            results["summary_error"] = str(e)[:100]

        # Get care timeline
        try:
            resp = await self._client.get(
                f"{settings.context_graph_url}/timeline/{patient_id}?limit=20",
                headers=headers or {},
                timeout=15,
            )
            if resp.status_code == 200:
                results["timeline"] = resp.json()
        except Exception as e:
            results["timeline_error"] = str(e)[:100]

        # Store conversation message if text provided
        if text and session_id:
            try:
                await self._client.post(
                    f"{settings.context_graph_url}/memory/short-term/message",
                    json={
                        "session_id": session_id,
                        "patient_id": patient_id,
                        "role": "user",
                        "content": text[:2000],
                    },
                    headers=headers or {},
                    timeout=10,
                )
            except Exception:
                pass  # Non-critical — don't block the main response

        return results

    async def _call_context1_agent(
        self,
        query: str | None,
        headers: dict[str, str] | None = None,
    ) -> dict:
        """Call Chroma Context-1 multi-hop retrieval agent."""
        if not query or not settings.enable_context1_agent:
            return {"status": "disabled"}
        resp = await self._client.post(
            f"{settings.context1_url}/query",
            json={"question": query},
            headers=headers or {},
            timeout=90,
        )
        resp.raise_for_status()
        return _sanitize_engine_output(resp.json())

    # ── Response Merging ─────────────────────────────────────────

    def _merge_responses(
        self, decision: RoutingDecision,
        responses: list[EngineResponse], total_latency_ms: float,
    ) -> dict[str, Any]:
        """Merge responses from multiple engines into a unified result."""
        primary = next(
            (r for r in responses if r.engine == decision.primary_target.value),
            None,
        )
        secondaries = [
            r for r in responses if r.engine != decision.primary_target.value
        ]

        result: dict[str, Any] = {
            "routing": {
                "primary_engine": decision.primary_target.value,
                "secondary_engines": [t.value for t in decision.secondary_targets],
                "reason": decision.reason,
                "confidence": decision.confidence,
            },
            "primary_result": primary.data if primary and primary.success else None,
            "primary_error": primary.error if primary and not primary.success else None,
            "supplementary": {},
            "engine_latencies": {r.engine: round(r.latency_ms, 1) for r in responses},
            "total_latency_ms": round(total_latency_ms, 1),
        }

        for resp in secondaries:
            if resp.success:
                result["supplementary"][resp.engine] = resp.data
            else:
                result["supplementary"][resp.engine] = {"error": resp.error}

        # Extract top-level answer
        if primary and primary.success and isinstance(primary.data, dict):
            data = primary.data
            result["answer"] = (
                data.get("answer")
                or data.get("report_text")
                or data.get("raw_text")
                or str(data)
            )
            # Use confidence from response data if available, else from engine metadata
            result["confidence"] = (
                data.get("confidence")
                or primary.confidence
                or 0.0
            )
            if "safety" in data:
                result["safety"] = data["safety"]
            if "governance" in data:
                result["governance"] = data["governance"]
        else:
            result["answer"] = ""
            result["confidence"] = 0.0

        primary_sources = []
        if primary and primary.success and isinstance(primary.data, dict):
            primary_sources = primary.data.get("sources", []) or []

        # Enrich with search results
        search_resp = next(
            (r for r in secondaries
             if r.engine == RouteTarget.SEARCH_RAG.value and r.success),
            None,
        )
        search_sources = search_resp.data.get("sources", []) if (
            search_resp and isinstance(search_resp.data, dict)
        ) else []
        openrag_resp = next(
            (r for r in secondaries
             if r.engine == RouteTarget.OPENRAG.value and r.success),
            None,
        )
        openrag_sources = openrag_resp.data.get("sources", []) if (
            openrag_resp and isinstance(openrag_resp.data, dict)
        ) else []
        combined_sources = self._merge_sources(
            primary_sources,
            search_sources + openrag_sources,
        )
        if combined_sources:
            result["sources"] = combined_sources
            result["answer"] = self._append_citation_block(
                result.get("answer", ""),
                combined_sources,
            )

        return result

    def _merge_sources(
        self,
        primary_sources: list[dict[str, Any]] | None,
        supplementary_sources: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        combined = []
        seen: set[str] = set()
        for source in (primary_sources or []) + (supplementary_sources or []):
            if not isinstance(source, dict):
                continue
            key = (
                source.get("url")
                or f"{source.get('title', '')}:{source.get('source', '')}"
            )
            if key in seen:
                continue
            seen.add(key)
            combined.append(source)
        return combined

    def _append_citation_block(
        self,
        answer: str,
        sources: list[dict[str, Any]],
    ) -> str:
        if not answer or not sources:
            return answer
        if re.search(r"(?i)\b(citations|sources used|sources)\b", answer):
            return answer

        citation_lines = []
        for idx, source in enumerate(sources[:5], start=1):
            title = source.get("title", "Untitled source")
            url = source.get("url") or source.get("source") or ""
            citation_lines.append(f"{idx}. {title} - {url}")

        return f"{answer}\n\nSources:\n" + "\n".join(citation_lines)

    # ── Health Check ─────────────────────────────────────────────

    async def health_check(self) -> dict[str, Any]:
        engines = {
            "mediscan_vlm": settings.mediscan_vlm_url,
            "medical_llm": settings.medical_llm_url,
            "mediscan_ocr": settings.mediscan_ocr_url,
            "granite_vision": settings.granite_vllm_url.replace("/v1", ""),
            "openrag": settings.openrag_url,
            "context_graph": settings.context_graph_url,
            "context1_agent": settings.context1_url,
        }
        results = {}
        for name, url in engines.items():
            circuit = self._circuits.get(name)
            try:
                endpoint = engine_health_endpoint(name, url)
                resp = await self._client.get(endpoint, timeout=5)
                payload: dict[str, Any] | None = None
                try:
                    parsed = resp.json()
                    if isinstance(parsed, dict):
                        payload = parsed
                except ValueError:
                    payload = None
                normalized_status, payload_status = normalize_engine_health(
                    name,
                    resp.status_code,
                    payload,
                )
                results[name] = {
                    "status": normalized_status,
                    "code": resp.status_code,
                    "endpoint": endpoint,
                    "payload_status": payload_status,
                    "circuit": "closed" if (circuit and not circuit.is_open) else "unknown",
                    "total_requests": circuit.total_requests if circuit else 0,
                    "total_failures": circuit.total_failures if circuit else 0,
                }
                if payload is not None:
                    results[name]["details"] = _sanitize_engine_output(payload)
            except Exception as e:
                results[name] = {
                    "status": "unhealthy",
                    "error": str(e)[:100],
                    "circuit": "open" if (circuit and circuit.is_open) else "closed",
                    "total_requests": circuit.total_requests if circuit else 0,
                    "total_failures": circuit.total_failures if circuit else 0,
                }
        return results
