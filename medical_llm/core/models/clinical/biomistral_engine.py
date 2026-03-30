"""
BioMistral-7B — Fast inference engine for routing and simple Q&A.
Lightweight, low-latency medical LLM for quick responses.
"""

from __future__ import annotations

import logging
from typing import Any

from core.models.base_model import HuggingFaceLLM

logger = logging.getLogger(__name__)


class BioMistralEngine(HuggingFaceLLM):
    """BioMistral-7B — fast, lightweight medical LLM.

    Used for:
    - Query routing decisions (classify query type)
    - Simple medical Q&A (factual questions)
    - Quick validation checks
    - First-pass screening before heavy models
    """

    SYSTEM_PROMPT = (
        "You are BioMistral, a precise and efficient medical AI assistant. "
        "Provide accurate, concise medical information. "
        "Be direct and factual. If a question requires deeper analysis, "
        "indicate that specialist evaluation is recommended."
    )

    def __init__(
        self,
        model_id: str = "BioMistral/BioMistral-7B",
        config: dict[str, Any] | None = None,
    ):
        super().__init__(model_id=model_id, config=config or {})

    def _build_chat_prompt(self, system_prompt: str, user_prompt: str) -> str:
        """Mistral instruct format."""
        return f"[INST] {system_prompt}\n\n{user_prompt} [/INST]"

    def quick_answer(
        self,
        query: str,
        max_new_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Fast medical Q&A.

        Args:
            query: Medical question

        Returns:
            Quick response dict
        """
        result = self.generate(
            query,
            system_prompt=self.SYSTEM_PROMPT,
            max_new_tokens=max_new_tokens or 1024,
        )

        return {
            "text": result.get("text", ""),
            "model": self.model_id,
            "role": "fast_engine",
            "tokens_generated": result.get("tokens_generated", 0),
            "latency": result.get("latency", 0),
        }

    def classify_query(self, query: str) -> dict[str, Any]:
        """Use BioMistral to classify ambiguous queries.

        Args:
            query: Query to classify

        Returns:
            Classification result with category
        """
        categories = [
            "diagnosis", "research", "simple_qa", "conversational",
            "drug_info", "treatment", "lab_interpretation", "emergency",
        ]

        prompt = (
            f"Classify this medical query into exactly one category.\n"
            f"Categories: {', '.join(categories)}\n\n"
            f"Query: {query}\n\n"
            f"Category:"
        )

        result = self.generate(prompt, max_new_tokens=32, temperature=0.1)
        response = result.get("text", "").strip().lower()

        # Parse category from response
        for cat in categories:
            if cat in response:
                return {"category": cat, "confidence": 0.8, "source": "biomistral"}

        return {"category": "simple_qa", "confidence": 0.4, "source": "biomistral_fallback"}
