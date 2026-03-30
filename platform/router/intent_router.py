"""
Multi-Layer Intent Router — Enterprise-grade query classification.

Routing architecture (inspired by ChatGPT/Claude/Gemini):

    ┌────────────────────────────────────────────┐
    │  LAYER 1: RULE ENGINE (FAST PATH, <1ms)    │
    │  - MIME type detection for files            │
    │  - Emergency keyword detection              │
    └────────────────────┬───────────────────────┘
                         ↓
    ┌────────────────────────────────────────────┐
    │  LAYER 2: EMBEDDING ROUTER (~5-10ms)       │
    │  - BGE-Large semantic similarity            │
    │  - Pre-computed intent vectors              │
    │  - Cosine similarity → top intents          │
    └────────────────────┬───────────────────────┘
                         ↓
    ┌────────────────────────────────────────────┐
    │  LAYER 3: CONFIDENCE RESOLVER              │
    │  - Combine rule + embedding scores          │
    │  - Multi-intent detection (DAG routing)     │
    │  - Fallback to LLM for ambiguous cases      │
    └────────────────────┬───────────────────────┘
                         ↓
    ┌────────────────────────────────────────────┐
    │  EXECUTION PLAN (parallel engine calls)     │
    └────────────────────────────────────────────┘

The embedding router uses pre-computed intent vectors for semantic
matching — no hardcoded patterns. A query like "tell me about" is
understood semantically, not matched against a string list.

Models required (from download script --tier core):
    - BAAI/bge-large-en-v1.5 (~1.3GB)
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── Intent Definitions ────────────────────────────────────────────────────
# Each intent maps to a route target and has representative descriptions
# that are embedded once at startup. The query is compared against these.

INTENT_DEFINITIONS: dict[str, dict[str, Any]] = {
    "general_conversation": {
        "route": "general_llm",
        "descriptions": [
            "hello, hi, hey, greetings, good morning",
            "who are you, what are you, tell me about yourself",
            "what can you do, how can you help me",
            "thank you, thanks, goodbye, bye, see you",
            "how are you, what's up, nice to meet you",
            "help me understand this platform",
            "I want to chat, let's talk",
            "tell me something interesting",
            "what is MedAI, describe your capabilities",
            "introduce yourself, your features",
        ],
    },
    "medical_question": {
        "route": "general_llm",  # General LLM with medical system prompt
        "descriptions": [
            "what are the symptoms of diabetes",
            "explain hypertension treatment options",
            "what causes chest pain",
            "side effects of metformin",
            "how to manage high blood pressure",
            "difference between type 1 and type 2 diabetes",
            "what is the treatment for pneumonia",
            "explain the stages of cancer",
            "what are the risk factors for heart disease",
            "how does insulin work in the body",
            "tell me about antibiotics for infection",
            "what medications treat depression",
            "explain the immune system",
            "how does chemotherapy work",
        ],
    },
    "image_analysis": {
        "route": "mediscan_vlm",
        "descriptions": [
            "analyze this chest X-ray",
            "what does this CT scan show",
            "interpret this MRI image",
            "look at this ultrasound",
            "check this mammogram for abnormalities",
            "analyze this pathology slide",
            "read this retinal scan",
            "examine this ECG image",
            "what do you see in this radiograph",
            "is there anything abnormal in this scan",
            "analyze this DICOM image",
            "review this medical image",
        ],
    },
    "document_processing": {
        "route": "mediscan_ocr",
        "descriptions": [
            "extract text from this lab report",
            "read this medical document",
            "process this prescription",
            "scan this medical form",
            "OCR this document",
            "what does my blood test report say",
            "read the values from this lab result",
            "extract information from this PDF",
            "digitize this handwritten prescription",
            "parse this medical record",
        ],
    },
    "web_search": {
        "route": "search_rag",
        "descriptions": [
            "latest research on cancer treatment",
            "recent studies about COVID-19",
            "current guidelines for diabetes management",
            "newest treatment for Alzheimer's",
            "what does the latest evidence say about",
            "search PubMed for clinical trials",
            "find recent medical literature on",
            "what are the 2024 2025 guidelines for",
            "latest recommendations from WHO",
            "new drug approvals for",
        ],
    },
    "patient_context": {
        "route": "patient_db",
        "descriptions": [
            "my previous test results",
            "what were my last lab values",
            "check my medical history",
            "my prescription records",
            "my past diagnoses",
            "show my health timeline",
            "review my patient records",
            "compare with my previous report",
            "my follow-up results",
        ],
    },
}

# Threshold for confident routing (below this → use general_llm as fallback)
_CONFIDENCE_THRESHOLD = 0.45
# Threshold for adding secondary intents
_SECONDARY_THRESHOLD = 0.35


class EmbeddingRouter:
    """
    Semantic intent router using pre-computed embedding vectors.

    Loads BGE-Large once at startup, embeds all intent descriptions,
    then routes new queries via cosine similarity in ~5-10ms.
    """

    def __init__(self) -> None:
        self._model = None
        self._intent_embeddings: dict[str, np.ndarray] = {}
        self._is_loaded = False

    def load(self, model_name: str = "BAAI/bge-large-en-v1.5") -> bool:
        """Load the embedding model and pre-compute intent vectors."""
        try:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding router model: %s", model_name)
            start = time.monotonic()

            self._model = SentenceTransformer(model_name)

            # Pre-compute intent embeddings (mean of all descriptions per intent)
            for intent_name, intent_def in INTENT_DEFINITIONS.items():
                descriptions = intent_def["descriptions"]
                embeddings = self._model.encode(
                    descriptions,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                # Store the mean embedding for each intent
                self._intent_embeddings[intent_name] = np.mean(embeddings, axis=0)

            elapsed = time.monotonic() - start
            self._is_loaded = True
            logger.info(
                "Embedding router loaded in %.1fs (%d intents, %d total descriptions)",
                elapsed, len(INTENT_DEFINITIONS),
                sum(len(d["descriptions"]) for d in INTENT_DEFINITIONS.values()),
            )
            return True

        except ImportError:
            logger.warning(
                "sentence-transformers not installed — embedding router disabled. "
                "Install with: pip install sentence-transformers"
            )
            return False
        except Exception as e:
            logger.error("Failed to load embedding router: %s", e)
            return False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def classify(self, query: str) -> list[tuple[str, float]]:
        """
        Classify a query into intents with confidence scores.

        Returns:
            List of (intent_name, confidence) tuples, sorted by confidence desc.
        """
        if not self._is_loaded or self._model is None:
            return []

        start = time.monotonic()

        # Embed the query
        query_embedding = self._model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        # Cosine similarity against all intent vectors
        scores: list[tuple[str, float]] = []
        for intent_name, intent_vec in self._intent_embeddings.items():
            similarity = float(np.dot(query_embedding, intent_vec))
            scores.append((intent_name, similarity))

        scores.sort(key=lambda x: x[1], reverse=True)

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.debug(
            "Embedding classification in %.1fms: query='%s' top=%s(%.3f)",
            elapsed_ms, query[:50], scores[0][0] if scores else "?",
            scores[0][1] if scores else 0,
        )

        return scores


class IntentRouter:
    """
    Multi-layer intent router combining rules + embeddings.

    Layer 1: Rule engine (MIME types, emergency keywords) — handled by MasterRouter
    Layer 2: Embedding router (semantic similarity) — this class
    Layer 3: Confidence resolver (combine scores, multi-intent) — this class
    """

    def __init__(self) -> None:
        self._embedding_router = EmbeddingRouter()

    def load(self, embedding_model: str = "BAAI/bge-large-en-v1.5") -> bool:
        """Initialize the embedding model."""
        return self._embedding_router.load(embedding_model)

    @property
    def is_loaded(self) -> bool:
        return self._embedding_router.is_loaded

    def route(self, query: str) -> dict[str, Any]:
        """
        Route a text query to the best engine(s).

        Returns:
            {
                "primary_intent": str,
                "primary_route": str,
                "primary_confidence": float,
                "secondary_intents": [{"intent": str, "route": str, "confidence": float}],
                "all_scores": [(intent, score), ...],
            }
        """
        if not self._embedding_router.is_loaded:
            # Fallback: route everything to general_llm
            return {
                "primary_intent": "general_conversation",
                "primary_route": "general_llm",
                "primary_confidence": 0.5,
                "secondary_intents": [],
                "all_scores": [],
                "method": "fallback",
            }

        scores = self._embedding_router.classify(query)

        if not scores:
            return {
                "primary_intent": "general_conversation",
                "primary_route": "general_llm",
                "primary_confidence": 0.5,
                "secondary_intents": [],
                "all_scores": [],
                "method": "fallback",
            }

        # Primary intent
        primary_intent, primary_score = scores[0]
        primary_route = INTENT_DEFINITIONS[primary_intent]["route"]

        # If confidence is below threshold, default to general_llm
        # (the LLM can handle anything naturally)
        if primary_score < _CONFIDENCE_THRESHOLD:
            primary_intent = "general_conversation"
            primary_route = "general_llm"
            primary_score = max(primary_score, 0.5)

        # Secondary intents (for multi-engine routing)
        secondary = []
        for intent_name, score in scores[1:]:
            if score >= _SECONDARY_THRESHOLD and intent_name != primary_intent:
                secondary.append({
                    "intent": intent_name,
                    "route": INTENT_DEFINITIONS[intent_name]["route"],
                    "confidence": round(score, 4),
                })

        return {
            "primary_intent": primary_intent,
            "primary_route": primary_route,
            "primary_confidence": round(primary_score, 4),
            "secondary_intents": secondary,
            "all_scores": [(name, round(s, 4)) for name, s in scores],
            "method": "embedding",
        }
