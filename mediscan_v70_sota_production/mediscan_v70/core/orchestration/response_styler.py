"""
MediScan AI v7.0 — Response Styler
Rewrites clinical model output into different conversational formats:
  - doctor:       Technical, structured, concise (for clinicians)
  - patient:      Simple, friendly, with emojis (for patients)
  - research:     Detailed, with reasoning chain (for researchers)
  - radiologist:  ACR-formatted structured reports
"""
from __future__ import annotations


import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ResponseStyler:
    """Rewrites raw clinical reports into mode-specific formats.

    Modes:
      doctor  → Technical, structured, ACR-compliant formatting
      patient → Simple language, friendly tone, emojis, avoid jargon
      research → Detailed with model reasoning, uncertainty metrics, citations
    """

    # Medical jargon → patient-friendly translations
    JARGON_MAP = {
        "opacity": "a cloudy area",
        "consolidation": "an area where the lung looks solid (possibly fluid or infection)",
        "effusion": "fluid buildup",
        "pleural effusion": "fluid around the lungs",
        "cardiomegaly": "an enlarged heart",
        "pneumothorax": "a collapsed lung (air leak)",
        "atelectasis": "a partially collapsed section of lung",
        "infiltrate": "an area that looks different from normal tissue",
        "nodule": "a small round spot",
        "mass": "an abnormal growth or lump",
        "edema": "swelling from fluid",
        "stenosis": "a narrowing",
        "fracture": "a break in the bone",
        "dislocation": "a bone moved out of position",
        "hemorrhage": "bleeding",
        "infarction": "tissue damage from blocked blood flow",
        "ischemia": "reduced blood flow",
        "fibrosis": "scarring",
        "necrosis": "dead tissue",
        "metastasis": "cancer that has spread",
        "malignant": "cancerous",
        "benign": "non-cancerous",
        "bilateral": "on both sides",
        "unilateral": "on one side",
        "proximal": "closer to the body center",
        "distal": "farther from the body center",
        "anterior": "front",
        "posterior": "back",
        "lateral": "side",
        "medial": "toward the middle",
        "parenchyma": "the main tissue of the organ",
        "periosteal": "the outer layer of bone",
        "cortical": "the outer part",
        "mediastinum": "the center of the chest",
        "hilum": "where blood vessels enter the lung",
        "costophrenic": "the area where the lung meets the diaphragm",
    }

    # Step-by-step display templates for patient mode
    ANALYSIS_STEPS = [
        ("🔍", "Looking at the overall image quality and orientation…"),
        ("🫁", "Examining the lung fields for any abnormalities…"),
        ("❤️", "Checking the heart size and shape…"),
        ("🦴", "Reviewing the bones and soft tissues…"),
        ("📋", "Compiling findings into a summary…"),
    ]

    def __init__(self):
        pass

    def rewrite(
        self,
        raw_report: dict[str, Any],
        mode: str = "patient",
        include_steps: bool = True,
    ) -> str:
        """Rewrite a clinical report into the specified mode.

        Args:
            raw_report: The structured report dict from ReportGenerator
            mode: "doctor", "patient", or "research"
            include_steps: If True and mode=patient, prepend step-by-step narrative

        Returns:
            Formatted text string
        """
        if mode == "patient":
            return self._rewrite_patient(raw_report, include_steps)
        elif mode == "research":
            return self._rewrite_research(raw_report)
        else:
            return self._rewrite_doctor(raw_report)

    def _rewrite_doctor(self, report: dict[str, Any]) -> str:
        """Doctor mode: Technical, structured, concise — ACR-style."""
        cr = report.get("clinical_report", {})
        gov = report.get("governance", {})
        ai = report.get("ai_metadata", {})
        risk = gov.get("risk_level", "routine")

        parts = []

        # Header
        parts.append(f"MEDISCAN AI v7.0 — DIAGNOSTIC REPORT")
        parts.append(f"Report ID: {report.get('report_id', 'N/A')}")
        parts.append(f"Date: {report.get('timestamp', 'N/A')}")
        parts.append(f"Risk: {risk.upper()}")
        parts.append("")

        # Critical alert
        if risk in ("emergent", "urgent"):
            parts.append(f"*** CRITICAL: {risk.upper()} findings — immediate attention required ***")
            parts.append("")

        # Sections
        for key, title in [("technique", "TECHNIQUE"), ("comparison", "COMPARISON"),
                           ("findings", "FINDINGS"), ("impression", "IMPRESSION"),
                           ("differential_diagnosis", "DIFFERENTIAL"),
                           ("recommendations", "RECOMMENDATIONS")]:
            val = cr.get(key, "")
            if val:
                parts.append(f"{title}:")
                parts.append(val)
                parts.append("")

        # Metadata
        parts.append(f"Models: {', '.join(ai.get('models_used', []))}")
        parts.append(f"Confidence: {ai.get('confidence', 0):.0%}")
        parts.append(f"Agreement: {ai.get('agreement_score', 0):.0%}")
        parts.append("")
        parts.append("DISCLAIMER: AI-generated. Must be reviewed by qualified physician.")

        return "\n".join(parts)

    def _rewrite_patient(self, report: dict[str, Any], include_steps: bool = True) -> str:
        """Patient mode: Simple, friendly, with emojis."""
        cr = report.get("clinical_report", {})
        gov = report.get("governance", {})
        ai = report.get("ai_metadata", {})
        risk = gov.get("risk_level", "routine")
        conf = ai.get("confidence", 0)

        parts = []

        # Step-by-step narrative
        if include_steps:
            parts.append("📸 Here's how I analyzed your image:\n")
            for icon, step in self.ANALYSIS_STEPS:
                parts.append(f"  {icon} {step}")
            parts.append("")

        # Confidence indicator
        if conf >= 0.8:
            parts.append("✅ I'm fairly confident in these results.\n")
        elif conf >= 0.5:
            parts.append("⚠️ These results have moderate confidence. Please discuss with your doctor.\n")
        else:
            parts.append("❗ These results are uncertain. A specialist should review them.\n")

        # Risk level
        risk_msgs = {
            "emergent": "🔴 IMPORTANT: The analysis found something that may need urgent attention. Please contact your doctor or visit the emergency room as soon as possible.",
            "urgent": "🟠 HEADS UP: Some findings need medical attention. Please schedule an appointment with your doctor soon.",
            "routine": "🟢 Overall, nothing appears to need immediate attention.",
        }
        parts.append(risk_msgs.get(risk, risk_msgs["routine"]))
        parts.append("")

        # Findings in simple language
        findings = cr.get("findings", "")
        if findings:
            parts.append("📋 What we found:")
            simple_findings = self._simplify_text(findings)
            parts.append(f"  {simple_findings}")
            parts.append("")

        # Impression
        impression = cr.get("impression", "")
        if impression:
            parts.append("💡 In simple terms:")
            simple_impression = self._simplify_text(impression)
            parts.append(f"  {simple_impression}")
            parts.append("")

        # Differential
        diff = cr.get("differential_diagnosis", "")
        if diff:
            parts.append("🤔 Possible explanations:")
            simple_diff = self._simplify_text(diff)
            parts.append(f"  {simple_diff}")
            parts.append("")

        # Recommendations
        recs = cr.get("recommendations", "")
        if recs:
            parts.append("📌 Next steps:")
            simple_recs = self._simplify_text(recs)
            parts.append(f"  {simple_recs}")
            parts.append("")

        # Disclaimer
        parts.append("─" * 50)
        parts.append("⚠️ Remember: This is an AI analysis, not a doctor's diagnosis.")
        parts.append("Always discuss these results with your healthcare provider.")

        return "\n".join(parts)

    def _rewrite_research(self, report: dict[str, Any]) -> str:
        """Research mode: Detailed with reasoning chain and metrics."""
        cr = report.get("clinical_report", {})
        gov = report.get("governance", {})
        ai = report.get("ai_metadata", {})

        parts = []

        parts.append("═══ MediScan AI v7.0 — Research Report ═══")
        parts.append(f"Report ID: {report.get('report_id', 'N/A')}")
        parts.append(f"Timestamp: {report.get('timestamp', 'N/A')}")
        parts.append("")

        # Model ensemble details
        parts.append("── Model Ensemble Analysis ──")
        individual = ai.get("individual_results", [])
        for ir in individual:
            m = ir.get("model", "unknown")
            c = ir.get("confidence", 0)
            w = ir.get("weight", 0.5)
            gen = "generative" if ir.get("is_generative") else "classifier"
            parts.append(f"  {m} [{gen}] — conf={c:.3f}, weight={w:.2f}")
            thinking = ir.get("thinking", "")
            if thinking:
                parts.append(f"    Reasoning chain: {thinking[:500]}")
            parts.append(f"    Output excerpt: {ir.get('excerpt', 'N/A')}")
            parts.append("")

        # Fusion metrics
        parts.append("── Fusion Metrics ──")
        parts.append(f"  Consensus confidence: {ai.get('confidence', 0):.4f}")
        parts.append(f"  Inter-model agreement: {ai.get('agreement_score', 0):.4f}")
        parts.append(f"  Uncertainty: {ai.get('uncertainty', 0):.4f}")
        parts.append(f"  Best model: {ai.get('best_model', 'N/A')}")
        parts.append("")

        # Clinical sections
        parts.append("── Clinical Sections ──")
        for key, title in [("technique", "Technique"), ("findings", "Findings"),
                           ("impression", "Impression"),
                           ("differential_diagnosis", "Differential Diagnosis"),
                           ("recommendations", "Recommendations")]:
            val = cr.get(key, "")
            if val:
                parts.append(f"\n  [{title}]")
                parts.append(f"  {val}")
        parts.append("")

        # Governance details
        parts.append("── Governance Layer ──")
        parts.append(f"  Risk level: {gov.get('risk_level', 'routine')}")
        parts.append(f"  Clinical valid: {gov.get('clinical_valid', False)}")
        critical = gov.get("critical_findings", [])
        if critical:
            parts.append(f"  Critical findings ({len(critical)}):")
            for cf in critical:
                parts.append(f"    - {cf.get('finding', '')} [{cf.get('urgency', '')}]")
        negated = gov.get("negated_findings", [])
        if negated:
            parts.append(f"  Negated findings ({len(negated)}):")
            for nf in negated:
                parts.append(f"    - {nf.get('finding', '')} (would be {nf.get('would_be_level', '')})")
        parts.append("")

        parts.append("═══ End Research Report ═══")
        return "\n".join(parts)

    def _simplify_text(self, text: str) -> str:
        """Replace medical jargon with patient-friendly language."""
        result = text
        # Sort by length descending to replace longer phrases first
        sorted_jargon = sorted(
            self.JARGON_MAP.items(), key=lambda x: len(x[0]), reverse=True
        )
        for jargon, simple in sorted_jargon:
            # Case-insensitive replacement
            pattern = re.compile(re.escape(jargon), re.IGNORECASE)
            result = pattern.sub(f"{simple} ({jargon})", result)
        return result
