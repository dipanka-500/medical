from __future__ import annotations

import os

from medicscan_ocr.backends.granite_vision import STRUCTURE_HEAVY_TYPES
from medicscan_ocr.config import Settings
from medicscan_ocr.schemas import AnalysisResult, DocumentType, LayoutComplexity, RoutingDecision

# Feature flag: enable/disable Granite Vision sidecar routing
_GRANITE_ENABLED = os.getenv("GRANITE_ENABLED", "true").lower() in {"1", "true", "yes"}


class RoutePlanner:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def _is_structure_heavy(self, analysis: AnalysisResult) -> bool:
        """Detect if the document is structure-heavy (tables, forms, KVP)."""
        hints = analysis.source_hints
        filename = hints.get("filename", "").lower()

        # Check filename keywords
        for doc_type in STRUCTURE_HEAVY_TYPES:
            keywords = doc_type.replace("_", " ").split()
            if any(kw in filename for kw in keywords):
                return True

        # Check if analysis detected tables or high layout complexity
        if analysis.needs_table_model:
            return True
        if analysis.layout_complexity == LayoutComplexity.HIGH:
            return True

        return False

    def decide(
        self,
        analysis: AnalysisResult,
        requested_backend: str = "auto",
    ) -> RoutingDecision:
        if requested_backend and requested_backend != "auto":
            return RoutingDecision(
                primary_backend=requested_backend,
                secondary_backends=[],
                enrichers=self._build_enrichers(analysis),
                reason=["User forced backend: {0}".format(requested_backend)],
                requested_backend=requested_backend,
            )

        reason = []
        enrichers = self._build_enrichers(analysis)
        secondary = []
        prefer_remote = self.settings.prefer_remote_api

        if analysis.document_type == DocumentType.HANDWRITTEN:
            if analysis.language_family == "english":
                primary = "chandra_command"
                secondary = ["surya_command"]
                if self.settings.has_sarvam_key:
                    secondary.append("sarvam_vision")
                reason.append("Handwritten English route selected Chandra first.")
            elif analysis.language_family == "indic":
                primary = "sarvam_vision" if self.settings.has_sarvam_key else "surya_command"
                secondary = ["chandra_command", "surya_command"] if self.settings.has_sarvam_key else ["chandra_command"]
                reason.append("Handwritten Indic route selected Sarvam first.")
                if not self.settings.has_sarvam_key:
                    reason.append("Sarvam API key is missing, so the route fell back to Surya locally.")
            else:
                primary = "chandra_command"
                secondary = ["surya_command"]
                if self.settings.has_sarvam_key:
                    secondary.append("sarvam_vision")
                reason.append("Handwritten language was uncertain, so the route starts with Chandra and keeps Indic-capable fallbacks.")
        elif analysis.document_type == DocumentType.PRINTED:
            if analysis.language_family == "english":
                primary = "firered_backend"
                secondary = ["surya_command", "doctr_placeholder", "donut_placeholder"]
                reason.append("Printed English route follows the FireRed-first design.")
            elif analysis.language_family == "indic":
                primary = "surya_command"
                secondary = ["doctr_placeholder", "donut_placeholder"]
                if self.settings.has_sarvam_key:
                    secondary.insert(0, "sarvam_vision")
                reason.append("Printed Indic route selected Surya first.")
            elif analysis.language_family in {"other", "mixed"}:
                primary = "surya_command"
                secondary = ["doctr_placeholder", "donut_placeholder"]
                reason.append("Printed multilingual route prefers Surya as the strongest local multilingual OCR backend.")
            else:
                primary = "firered_backend"
                secondary = ["surya_command", "doctr_placeholder"]
                reason.append("Unknown printed language starts with FireRed for Latin-heavy documents, with Surya as multilingual fallback.")
        else:
            if analysis.language_family == "indic":
                primary = "sarvam_vision" if self.settings.has_sarvam_key else "surya_command"
                secondary = ["chandra_command", "surya_command"] if self.settings.has_sarvam_key else ["chandra_command"]
                reason.append("Mixed or unknown Indic documents keep the Indic-first path.")
            else:
                primary = "chandra_command" if analysis.handwritten_confidence >= 0.50 else "firered_backend"
                secondary = ["surya_command", "doctr_placeholder"]
                if self.settings.has_sarvam_key:
                    secondary.append("sarvam_vision")
                reason.append("Mixed or unknown non-Indic documents are routed by handwriting confidence.")

        if prefer_remote and self.settings.has_sarvam_key and primary != "sarvam_vision":
            if "sarvam_vision" in secondary:
                secondary.remove("sarvam_vision")
            secondary.insert(0, primary)
            primary = "sarvam_vision"
            reason.append("prefer_remote_api promoted Sarvam Vision to primary.")

        if analysis.needs_formula_model and "donut_placeholder" not in secondary:
            secondary.append("donut_placeholder")
            reason.append("Formula-heavy filename hints added Donut as a fallback reviewer.")

        if analysis.layout_complexity == LayoutComplexity.HIGH and "pixtral_placeholder" not in secondary:
            secondary.append("pixtral_placeholder")
            reason.append("High layout complexity added Pixtral as a contradiction-review model.")

        # ── Granite Vision: add as secondary for structure-heavy documents ──
        if _GRANITE_ENABLED and self._is_structure_heavy(analysis):
            if "granite_vision" not in secondary and primary != "granite_vision":
                secondary.insert(0, "granite_vision")
                reason.append(
                    "Structure-heavy document detected — Granite Vision 4.0 3B added "
                    "for table extraction, KVP extraction, and contradiction review."
                )

        return RoutingDecision(
            primary_backend=primary,
            secondary_backends=secondary,
            enrichers=enrichers,
            reason=reason,
            requested_backend=None,
        )

    def _build_enrichers(self, analysis: AnalysisResult) -> list:
        enrichers = []
        if analysis.needs_layout_model:
            enrichers.append("layoutlmv3_placeholder")
        if analysis.needs_table_model:
            enrichers.append("table_transformer_placeholder")
        return enrichers
