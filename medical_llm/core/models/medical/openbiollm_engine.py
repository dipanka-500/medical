"""
OpenBioLLM-70B — SOTA open-source medical LLM.
Primary clinical reasoning alongside DeepSeek-R1.
"""

from __future__ import annotations

import logging
from typing import Any

from core.models.base_model import HuggingFaceLLM

logger = logging.getLogger(__name__)


class OpenBioLLMEngine(HuggingFaceLLM):
    """OpenBioLLM-70B — state-of-the-art open medical LLM.

    Built on Llama-3 architecture, fine-tuned on high-quality medical
    datasets. Excels at clinical reasoning, medical Q&A, and diagnosis.
    """

    SYSTEM_PROMPT = (
        "You are OpenBioLLM, an expert medical AI assistant. "
        "You combine comprehensive biomedical knowledge with clinical reasoning. "
        "Follow these principles:\n"
        "1. Provide thorough, evidence-based responses\n"
        "2. Structure your answers clearly with relevant sections\n"
        "3. Include confidence levels for diagnoses and recommendations\n"
        "4. Reference medical guidelines when applicable\n"
        "5. Flag any safety concerns or red flags immediately\n"
        "6. Consider all relevant differential diagnoses\n"
        "7. Recommend appropriate follow-up when needed"
    )

    def __init__(
        self,
        model_id: str = "aaditya/Llama3-OpenBioLLM-70B",
        config: dict[str, Any] | None = None,
    ):
        super().__init__(model_id=model_id, config=config or {})

    def _build_chat_prompt(self, system_prompt: str, user_prompt: str) -> str:
        """Llama-3 chat format."""
        return (
            f"<|begin_of_text|>"
            f"<|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_prompt}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_prompt}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    def analyze(
        self,
        query: str,
        rag_context: str = "",
        max_new_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Run clinical analysis with OpenBioLLM.

        Args:
            query: Medical query or clinical presentation
            rag_context: Retrieved evidence

        Returns:
            Clinical analysis result
        """
        prompt_parts = []

        if rag_context:
            prompt_parts.append(f"## Evidence from Medical Literature\n{rag_context}\n")

        prompt_parts.append(f"## Clinical Query\n{query}\n")
        prompt_parts.append(
            "## Instructions\n"
            "Provide a comprehensive clinical analysis including:\n"
            "1. Clinical assessment and key findings\n"
            "2. Differential diagnosis with probabilities\n"
            "3. Supporting evidence for each diagnosis\n"
            "4. Recommended investigations\n"
            "5. Management plan\n"
            "6. Safety considerations and red flags\n"
            "7. Confidence level for your overall assessment"
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
            "role": "medical_reasoning",
            "tokens_generated": result.get("tokens_generated", 0),
            "latency": result.get("latency", 0),
        }
