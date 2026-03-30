"""
Med42-70B — Secondary diagnostic validator.
Cross-checks diagnoses and drug recommendations.
"""

from __future__ import annotations

import logging
from typing import Any

from core.models.base_model import HuggingFaceLLM

logger = logging.getLogger(__name__)


class Med42Engine(HuggingFaceLLM):
    """Med42 — secondary medical validator and diagnostic cross-checker.

    Specializes in:
    - Cross-checking diagnoses from other models
    - Verifying drug recommendations and interactions
    - Providing independent clinical assessment for consensus
    """

    SYSTEM_PROMPT = (
        "You are Med42, an advanced medical AI designed for clinical validation. "
        "You provide independent assessments to cross-check other clinical analyses. "
        "Focus on:\n"
        "1. Accuracy of diagnoses and differential diagnoses\n"
        "2. Appropriateness of recommended treatments\n"
        "3. Drug safety and interaction checking\n"
        "4. Consistency with current evidence and guidelines\n"
        "5. Identifying potential oversights or biases"
    )

    CROSSCHECK_TEMPLATE = (
        "## Clinical Scenario\n{query}\n\n"
        "## Previous Analysis to Cross-Check\n{analysis}\n\n"
        "## Your Independent Assessment\n"
        "Provide your own independent clinical analysis, then compare it with "
        "the previous analysis above. Note any:\n"
        "1. Areas of agreement (increases confidence)\n"
        "2. Areas of disagreement (needs review)\n"
        "3. Missing considerations\n"
        "4. Drug safety concerns\n"
        "5. Your overall agreement level (AGREE / PARTIALLY_AGREE / DISAGREE)"
    )

    def __init__(
        self,
        model_id: str = "m42-health/Llama3-Med42-70B",
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

    def cross_check(
        self,
        query: str,
        analysis: str,
        max_new_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Cross-check another model's clinical analysis.

        Args:
            query: Original clinical query
            analysis: Output from primary model to verify

        Returns:
            Cross-check result with agreement assessment
        """
        prompt = self.CROSSCHECK_TEMPLATE.format(
            query=query,
            analysis=analysis,
        )

        result = self.generate(
            prompt,
            system_prompt=self.SYSTEM_PROMPT,
            max_new_tokens=max_new_tokens or 2048,
        )

        response_text = result.get("text", "")

        # Parse agreement level
        agreement = "PARTIALLY_AGREE"
        response_upper = response_text.upper()
        if "DISAGREE" in response_upper and "PARTIALLY" not in response_upper:
            agreement = "DISAGREE"
        elif "AGREE" in response_upper and "PARTIALLY" not in response_upper and "DISAGREE" not in response_upper:
            agreement = "AGREE"

        return {
            "text": response_text,
            "agreement": agreement,
            "model": self.model_id,
            "role": "validator",
            "tokens_generated": result.get("tokens_generated", 0),
            "latency": result.get("latency", 0),
        }

    def verify_medications(
        self,
        medications: list[str],
        patient_context: str = "",
        max_new_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Verify medication recommendations for safety.

        Args:
            medications: List of medications to check
            patient_context: Patient info for interaction checking

        Returns:
            Medication verification result
        """
        med_list = "\n".join(f"- {m}" for m in medications)
        prompt = (
            f"## Medications to Verify\n{med_list}\n\n"
            f"{'## Patient Context: ' + patient_context if patient_context else ''}\n\n"
            f"## Task\n"
            f"For each medication, verify:\n"
            f"1. Normal dosage ranges\n"
            f"2. Common contraindications\n"
            f"3. Drug-drug interactions between listed medications\n"
            f"4. Common side effects patients should be warned about\n"
            f"5. Any black box warnings\n\n"
            f"Flag any safety concerns with ⚠️ markers."
        )

        result = self.generate(
            prompt,
            system_prompt=self.SYSTEM_PROMPT,
            max_new_tokens=max_new_tokens or 2048,
        )

        return {
            "text": result.get("text", ""),
            "model": self.model_id,
            "role": "drug_validator",
            "tokens_generated": result.get("tokens_generated", 0),
            "latency": result.get("latency", 0),
        }
