"""
Me-LLaMA — Conversational medical Q&A engine.
Optimized for patient-friendly explanations and medical dialogue.
"""

from __future__ import annotations

import logging
from typing import Any

from core.models.base_model import HuggingFaceLLM

logger = logging.getLogger(__name__)


class MeLLaMAEngine(HuggingFaceLLM):
    """Me-LLaMA conversational medical engine.

    Strengths:
    - Patient-friendly language
    - Multi-turn medical dialogue
    - Plain-language explanations of complex medical concepts
    - Empathetic communication style
    """

    SYSTEM_PROMPT = (
        "You are Me-LLaMA, a compassionate and knowledgeable medical AI assistant. "
        "Your role is to explain medical concepts clearly and empathetically. "
        "Always:\n"
        "1. Use plain, accessible language that patients can understand\n"
        "2. Explain medical terms when you must use them\n"
        "3. Be empathetic and supportive in tone\n"
        "4. Provide context for why things matter\n"
        "5. Recommend consulting healthcare providers for personalized advice\n"
        "6. Never diagnose — instead, explain possibilities and next steps\n"
        "7. Include relevant lifestyle and preventive care advice when appropriate"
    )

    def __init__(
        self,
        model_id: str = "clinicalnlplab/Me-LLaMA-13b",
        config: dict[str, Any] | None = None,
    ):
        super().__init__(model_id=model_id, config=config or {})

    def _build_chat_prompt(self, system_prompt: str, user_prompt: str) -> str:
        """Me-LLaMA prompt format."""
        return (
            f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
            f"{user_prompt} [/INST]"
        )

    def answer(
        self,
        query: str,
        conversation_history: list[dict[str, str]] | None = None,
        rag_context: str = "",
        max_new_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Generate a conversational medical response.

        Args:
            query: Patient's question
            conversation_history: Previous turns [{role, content}, ...]
            rag_context: Retrieved evidence

        Returns:
            Conversational response dict
        """
        # Build prompt with history
        prompt_parts = []

        if rag_context:
            prompt_parts.append(f"Reference information:\n{rag_context}\n")

        if conversation_history:
            prompt_parts.append("Previous conversation:")
            for turn in conversation_history[-6:]:  # Last 6 turns
                role = turn.get("role", "user")
                content = turn.get("content", "")
                prefix = "Patient" if role == "user" else "Assistant"
                prompt_parts.append(f"{prefix}: {content}")
            prompt_parts.append("")

        prompt_parts.append(f"Patient's question: {query}")
        prompt_parts.append(
            "\nPlease provide a clear, empathetic response. "
            "Explain any medical terms in simple language."
        )

        full_prompt = "\n".join(prompt_parts)

        result = self.generate(
            full_prompt,
            system_prompt=self.SYSTEM_PROMPT,
            max_new_tokens=max_new_tokens,
        )

        return {
            "text": result.get("text", ""),
            "model": self.model_id,
            "role": "conversational",
            "tokens_generated": result.get("tokens_generated", 0),
            "latency": result.get("latency", 0),
        }
