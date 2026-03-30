"""
MediScan AI v5.0 — Medical Reasoning Engine
Chain-of-thought medical reasoning using MediX-R1 + differential diagnosis.
"""
from __future__ import annotations


import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class MedicalReasoner:
    """Orchestrates multi-step medical reasoning across model outputs."""

    # Structured reasoning prompts for different clinical scenarios
    REASONING_PROMPTS = {
        "diagnosis": (
            "Based on the imaging findings, please provide:\n"
            "1. Primary diagnosis with confidence level\n"
            "2. Differential diagnoses (at least 3)\n"
            "3. Supporting evidence from the image\n"
            "4. Recommended follow-up studies\n"
            "5. Clinical significance and urgency level\n\n"
            "Think step by step about each finding."
        ),
        "report": (
            "Generate a comprehensive radiology report with:\n"
            "1. TECHNIQUE: Imaging modality and parameters\n"
            "2. COMPARISON: Note if comparison studies available\n"
            "3. FINDINGS: Systematic description of all findings\n"
            "4. IMPRESSION: Summary of key findings\n"
            "5. RECOMMENDATIONS: Suggested follow-up\n\n"
            "Be thorough and systematic."
        ),
        "differential": (
            "Analyze this medical image and provide a differential diagnosis:\n"
            "1. List the top 5 most likely diagnoses\n"
            "2. For each diagnosis, explain:\n"
            "   - Key imaging features supporting it\n"
            "   - Key imaging features against it\n"
            "   - Estimated probability\n"
            "3. Recommend the most appropriate next diagnostic step\n\n"
            "Reason carefully through each possibility."
        ),
        "surgical": (
            "Analyze this surgical/procedural image or video:\n"
            "1. Identify the surgical phase/step\n"
            "2. Identify visible anatomical structures\n"
            "3. Note any abnormalities or complications\n"
            "4. Assess surgical technique quality\n"
            "5. Provide safety observations\n\n"
            "Be precise and methodical."
        ),
    }

    def build_reasoning_prompt(
        self,
        user_query: str,
        model_outputs: list[dict[str, Any]],
        scenario: str = "diagnosis",
    ) -> str:
        """Build a chain-of-thought reasoning prompt that incorporates
        outputs from primary models.

        Args:
            user_query: Original user question
            model_outputs: Results from primary/verifier models
            scenario: Type of clinical reasoning
        """
        base_prompt = self.REASONING_PROMPTS.get(scenario, self.REASONING_PROMPTS["diagnosis"])

        # Synthesize model outputs into context
        context_parts = []
        for i, output in enumerate(model_outputs, 1):
            model_name = output.get("model", f"Model {i}")
            response = output.get("answer", output.get("response", ""))
            confidence = output.get("confidence", "N/A")

            context_parts.append(
                f"### Analysis from {model_name} (confidence: {confidence}):\n{response}"
            )

        context = "\n\n".join(context_parts)

        prompt = (
            f"You are an expert medical AI performing clinical reasoning.\n\n"
            f"## Original Question\n{user_query}\n\n"
            f"## Preliminary Analyses\n{context}\n\n"
            f"## Your Task\n{base_prompt}\n\n"
            f"Synthesize the above analyses and provide your expert assessment. "
            f"If the analyses conflict, explain why and provide your best judgment."
        )

        return prompt

    def extract_structured_reasoning(self, response: str) -> dict[str, Any]:
        """Parse a reasoning response into structured components."""
        sections = {
            "primary_diagnosis": "",
            "differential_diagnoses": [],
            "supporting_evidence": [],
            "recommendations": [],
            "urgency": "routine",
            "confidence": 0.0,
        }

        lines = response.split("\n")
        current_section = None

        for line in lines:
            line_lower = line.lower().strip()

            if any(kw in line_lower for kw in ["primary diagnosis", "impression", "assessment"]):
                current_section = "primary_diagnosis"
            elif any(kw in line_lower for kw in ["differential", "possibilities"]):
                current_section = "differential"
            elif any(kw in line_lower for kw in ["evidence", "finding", "support"]):
                current_section = "evidence"
            elif any(kw in line_lower for kw in ["recommend", "follow-up", "next step"]):
                current_section = "recommendations"
            elif any(kw in line_lower for kw in ["urgent", "emergent", "critical", "stat"]):
                sections["urgency"] = "urgent"
            elif line.strip().startswith(("-", "•", "*")) or line.strip()[:2].rstrip(". ").isdigit():
                content = line.strip().lstrip("-•* 0123456789.)")
                if current_section == "primary_diagnosis":
                    sections["primary_diagnosis"] = content
                elif current_section == "differential":
                    sections["differential_diagnoses"].append(content)
                elif current_section == "evidence":
                    sections["supporting_evidence"].append(content)
                elif current_section == "recommendations":
                    sections["recommendations"].append(content)

        return sections


class DifferentialDiagnosis:
    """Generates structured differential diagnoses from model outputs."""

    def generate(
        self,
        findings: list[str],
        modality: str,
        body_part: str = "",
        patient_context: str = "",
    ) -> dict[str, Any]:
        """Generate a differential diagnosis structure.

        This provides a template that gets populated by the reasoning model.
        """
        prompt = (
            f"Given the following imaging findings from a {modality} of "
            f"{'the ' + body_part if body_part else 'unknown body region'}:\n\n"
        )

        for i, finding in enumerate(findings, 1):
            prompt += f"{i}. {finding}\n"

        if patient_context:
            prompt += f"\nPatient Context: {patient_context}\n"

        prompt += (
            "\nProvide a ranked differential diagnosis with:\n"
            "- Diagnosis name\n"
            "- Probability estimate (%)\n"
            "- Key supporting features\n"
            "- Features that argue against\n"
            "- Recommended confirmatory test\n"
        )

        return {
            "prompt": prompt,
            "findings": findings,
            "modality": modality,
            "body_part": body_part,
        }
