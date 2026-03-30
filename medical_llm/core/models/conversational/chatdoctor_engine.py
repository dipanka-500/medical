"""
ChatDoctor — Patient-facing conversational medical assistant.
"""

from __future__ import annotations

import logging
from typing import Any

from core.models.base_model import HuggingFaceLLM

logger = logging.getLogger(__name__)


class ChatDoctorEngine(HuggingFaceLLM):
    """ChatDoctor — patient-focused medical conversation AI.

    Designed for empathetic, accessible medical dialogue.
    Handles patient Q&A, follow-up discussions, and health education.
    """

    SYSTEM_PROMPT = (
        "You are ChatDoctor, a friendly and knowledgeable medical AI assistant. "
        "You help patients understand their health concerns in simple, clear language. "
        "Guidelines:\n"
        "1. Be warm, empathetic, and patient\n"
        "2. Use simple language — avoid jargon unless explaining it\n"
        "3. Provide helpful health information and education\n"
        "4. Always recommend consulting a healthcare provider for specific medical advice\n"
        "5. Never make definitive diagnoses\n"
        "6. Ask clarifying questions when the situation is unclear\n"
        "7. Provide emotional support when discussing concerning symptoms"
    )

    def __init__(
        self,
        model_id: str = "zl111/ChatDoctor",
        config: dict[str, Any] | None = None,
    ):
        super().__init__(model_id=model_id, config=config or {})

    def chat(
        self,
        message: str,
        conversation_history: list[dict[str, str]] | None = None,
        max_new_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Have a conversation with the patient.

        Args:
            message: Patient's message
            conversation_history: Previous turns

        Returns:
            Conversational response
        """
        prompt_parts = []

        if conversation_history:
            for turn in conversation_history[-8:]:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                if role == "user":
                    prompt_parts.append(f"Patient: {content}")
                else:
                    prompt_parts.append(f"ChatDoctor: {content}")
            prompt_parts.append("")

        prompt_parts.append(f"Patient: {message}")
        prompt_parts.append("ChatDoctor:")

        full_prompt = "\n".join(prompt_parts)

        result = self.generate(
            full_prompt,
            system_prompt=self.SYSTEM_PROMPT,
            max_new_tokens=max_new_tokens or 1024,
        )

        return {
            "text": result.get("text", ""),
            "model": self.model_id,
            "role": "conversational",
            "tokens_generated": result.get("tokens_generated", 0),
            "latency": result.get("latency", 0),
        }
