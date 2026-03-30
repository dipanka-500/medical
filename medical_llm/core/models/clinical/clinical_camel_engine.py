"""
Clinical Camel — Clinical safety validator.
Reviews outputs for clinical accuracy and safety.
"""

from __future__ import annotations

import logging
from typing import Any

from core.models.base_model import HuggingFaceLLM

logger = logging.getLogger(__name__)


class ClinicalCamelEngine(HuggingFaceLLM):
    """Clinical Camel — safety and accuracy validator.

    Used as a VERIFICATION model, not primary inference.
    Reviews clinical outputs from other models and flags:
    - Factual medical errors
    - Unsafe recommendations
    - Missing safety warnings
    - Guideline non-compliance
    """

    VALIDATION_SYSTEM_PROMPT = (
        "You are ClinicalCamel, a safety-focused medical AI validator. "
        "Your sole purpose is to REVIEW and VALIDATE medical analyses generated "
        "by other AI models. You must:\n"
        "1. Identify factual medical errors\n"
        "2. Flag unsafe or potentially harmful recommendations\n"
        "3. Check for missing safety warnings or red flags\n"
        "4. Verify guideline compliance\n"
        "5. Assess dosage accuracy for any medications mentioned\n"
        "6. Check for drug interaction risks\n"
        "7. Ensure appropriate disclaimers are present\n\n"
        "Respond with a structured validation report."
    )

    VALIDATION_TEMPLATE = (
        "## Original Query\n{query}\n\n"
        "## Model Output to Validate\n{model_output}\n\n"
        "## Validation Checklist\n"
        "Review the above model output and provide:\n\n"
        "### 1. Factual Accuracy\n"
        "- Are all medical facts correct? List any errors.\n\n"
        "### 2. Safety Assessment\n"
        "- Are there any unsafe recommendations?\n"
        "- Are drug dosages within safe ranges?\n"
        "- Are contraindications addressed?\n\n"
        "### 3. Completeness\n"
        "- Are important diagnoses missing from the differential?\n"
        "- Are critical safety warnings included?\n\n"
        "### 4. Guideline Compliance\n"
        "- Does the output align with current medical guidelines?\n\n"
        "### 5. Overall Assessment\n"
        "- SAFE / NEEDS_REVIEW / UNSAFE\n"
        "- Confidence in assessment: HIGH / MEDIUM / LOW"
    )

    def __init__(
        self,
        model_id: str = "wanglab/ClinicalCamel-70B",
        config: dict[str, Any] | None = None,
    ):
        super().__init__(model_id=model_id, config=config or {})

    def validate(
        self,
        query: str,
        model_output: str,
        max_new_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Validate a model's clinical output for safety and accuracy.

        Args:
            query: Original medical query
            model_output: Output from another model to validate

        Returns:
            Validation result with safety assessment
        """
        prompt = self.VALIDATION_TEMPLATE.format(
            query=query,
            model_output=model_output,
        )

        result = self.generate(
            prompt,
            system_prompt=self.VALIDATION_SYSTEM_PROMPT,
            max_new_tokens=max_new_tokens or 2048,
        )

        validation_text = result.get("text", "")

        # Parse safety status
        safety_status = "NEEDS_REVIEW"
        if "UNSAFE" in validation_text.upper():
            safety_status = "UNSAFE"
        elif "SAFE" in validation_text.upper() and "UNSAFE" not in validation_text.upper():
            safety_status = "SAFE"

        return {
            "text": validation_text,
            "safety_status": safety_status,
            "is_safe": safety_status == "SAFE",
            "model": self.model_id,
            "role": "validator",
            "tokens_generated": result.get("tokens_generated", 0),
            "latency": result.get("latency", 0),
        }
