"""
Meditron-70B — Clinical reasoning specialist.
Structured clinical prompting with differential diagnosis templates.
"""

from __future__ import annotations

import logging
from typing import Any

from core.models.base_model import HuggingFaceLLM

logger = logging.getLogger(__name__)


class MeditronEngine(HuggingFaceLLM):
    """Meditron-70B clinical reasoning engine.

    Specialized for structured clinical reasoning with:
    - Systematic differential diagnosis
    - Evidence-based clinical decision support
    - Guideline-referenced recommendations
    """

    CLINICAL_SYSTEM_PROMPT = (
        "You are Meditron, a clinical reasoning AI trained on medical guidelines, "
        "PubMed literature, and clinical practice data. Your role is to provide:\n"
        "1. Systematic clinical assessments\n"
        "2. Evidence-based differential diagnoses\n"
        "3. Guideline-aligned treatment recommendations\n"
        "4. Risk stratification for clinical findings\n\n"
        "Always cite relevant guidelines (WHO, NICE, ACR, AHA) when applicable. "
        "Be precise, thorough, and prioritize patient safety."
    )

    DIFFERENTIAL_TEMPLATE = (
        "## Clinical Presentation\n{query}\n\n"
        "{rag_context}"
        "## Task: Comprehensive Clinical Analysis\n"
        "Provide the following:\n\n"
        "### A. Assessment\n"
        "- Chief complaint and key findings\n"
        "- Relevant positive and negative findings\n\n"
        "### B. Differential Diagnosis (ranked by probability)\n"
        "For each diagnosis:\n"
        "- Diagnosis name and ICD-10 code if applicable\n"
        "- Estimated probability (%)\n"
        "- Supporting evidence from the presentation\n"
        "- Evidence against\n"
        "- Key distinguishing features\n\n"
        "### C. Recommended Workup\n"
        "- Laboratory studies\n"
        "- Imaging studies\n"
        "- Special tests or procedures\n\n"
        "### D. Initial Management\n"
        "- Immediate interventions (if urgent)\n"
        "- First-line treatment options\n"
        "- Monitoring parameters\n\n"
        "### E. Red Flags\n"
        "- Findings that require immediate action\n"
        "- Conditions that must not be missed"
    )

    def __init__(
        self,
        model_id: str = "epfl-llm/meditron-70b",
        config: dict[str, Any] | None = None,
    ):
        super().__init__(model_id=model_id, config=config or {})

    def _build_chat_prompt(self, system_prompt: str, user_prompt: str) -> str:
        """Meditron chat format (Llama-style)."""
        return (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def clinical_analysis(
        self,
        query: str,
        rag_context: str = "",
        max_new_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Run structured clinical analysis.

        Args:
            query: Patient presentation or clinical question
            rag_context: Retrieved evidence

        Returns:
            Dict with structured clinical analysis
        """
        rag_section = f"## Retrieved Evidence\n{rag_context}\n\n" if rag_context else ""

        prompt = self.DIFFERENTIAL_TEMPLATE.format(
            query=query,
            rag_context=rag_section,
        )

        result = self.generate(
            prompt,
            system_prompt=self.CLINICAL_SYSTEM_PROMPT,
            max_new_tokens=max_new_tokens,
        )

        return {
            "text": result.get("text", ""),
            "model": self.model_id,
            "role": "medical_reasoning",
            "tokens_generated": result.get("tokens_generated", 0),
            "latency": result.get("latency", 0),
        }
