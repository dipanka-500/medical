from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image

from medicscan_ocr.config import Settings

logger = logging.getLogger(__name__)
from medicscan_ocr.models.handwriting import HandwritingClassifier
from medicscan_ocr.models.language import HandwritingLanguageDetector
from medicscan_ocr.schemas import AnalysisResult, DocumentType, LayoutComplexity, PreparedDocument
from medicscan_ocr.utils.files import is_pdf_path


INDIC_PREFIXES = (
    "hi",
    "ta",
    "te",
    "kn",
    "ml",
    "mr",
    "gu",
    "pa",
    "bn",
    "or",
    "as",
    "ur",
)


def _normalize_language(language_hint: Optional[str], default_language: str) -> Tuple[str, str, float]:
    if not language_hint:
        return "unknown", "unknown", 0.0
    value = language_hint.strip()
    prefix = value.split("-")[0].lower()
    if prefix in {"en", "eng", "english"}:
        return "en-IN", "english", 0.95
    if prefix in INDIC_PREFIXES:
        return value, "indic", 0.95
    if prefix in {"mixed", "multi"}:
        return default_language, "mixed", 0.60
    return value, "other", 0.80


def _layout_from_name(path: Path) -> Tuple[LayoutComplexity, bool, bool]:
    name = path.stem.lower()
    needs_table_model = any(
        token in name for token in ("table", "invoice", "ledger", "statement", "grid")
    )
    needs_formula_model = any(
        token in name for token in ("math", "equation", "formula", "chem")
    )
    if needs_table_model or needs_formula_model:
        return LayoutComplexity.HIGH, needs_table_model, needs_formula_model
    if any(token in name for token in ("form", "scan", "multicol", "paper")):
        return LayoutComplexity.MEDIUM, False, False
    return LayoutComplexity.LOW, False, False


class DocumentIntelligenceAnalyzer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.handwriting = HandwritingClassifier(
            checkpoint_path=str(settings.handwriting_checkpoint),
            handwritten_index=settings.handwriting_index,
        )
        self.handwriting_language = HandwritingLanguageDetector(
            default_indic_language_code=settings.default_indic_language_code
        )

    def analyze(
        self,
        prepared_document: PreparedDocument,
        language_hint: Optional[str] = None,
        document_type_hint: Optional[str] = None,
    ) -> AnalysisResult:
        path = Path(prepared_document.input_path).resolve()
        warnings = []
        source_hints = {
            "prepared_source_type": prepared_document.source_type,
            "prepared_page_count": len(prepared_document.pages),
        }
        if prepared_document.warnings:
            warnings.extend(prepared_document.warnings)

        layout_complexity, needs_table_model, needs_formula_model = _layout_from_name(path)
        needs_layout_model = layout_complexity != LayoutComplexity.LOW

        analysis_pages = prepared_document.pages[: self.settings.analysis_page_limit]
        page_paths = [page.processed_path for page in analysis_pages]

        if document_type_hint:
            normalized = document_type_hint.strip().lower()
            if normalized in {"printed", "handwritten", "mixed"}:
                document_type = DocumentType(normalized)
            else:
                document_type = DocumentType.UNKNOWN
                warnings.append("Unsupported document_type_hint: {0}".format(document_type_hint))
            handwritten_confidence = 1.0 if document_type == DocumentType.HANDWRITTEN else 0.0
            source_hints["document_type_source"] = "user_hint"
        elif page_paths:
            probabilities = []
            classifier_sources = []
            for page_path in page_paths:
                classification = self.handwriting.classify(
                    page_path,
                    high_threshold=self.settings.handwritten_high_threshold,
                    low_threshold=self.settings.handwritten_low_threshold,
                )
                probabilities.append(float(classification["confidence"]))
                classifier_sources.append(classification["source"])
            handwritten_confidence = sum(probabilities) / len(probabilities)
            if handwritten_confidence >= self.settings.handwritten_high_threshold:
                document_type = DocumentType.HANDWRITTEN
            elif handwritten_confidence <= self.settings.handwritten_low_threshold:
                document_type = DocumentType.PRINTED
            else:
                document_type = DocumentType.MIXED
            source_hints["document_type_source"] = classifier_sources
        elif is_pdf_path(path):
            document_type = DocumentType.UNKNOWN
            handwritten_confidence = 0.5
            warnings.append(
                "Page images were not available for this PDF, so handwriting detection stayed uncertain."
            )
            source_hints["document_type_source"] = "page_images_unavailable"
        else:
            document_type = DocumentType.UNKNOWN
            handwritten_confidence = 0.5
            source_hints["document_type_source"] = "unknown"

        language_code, language_family, language_confidence = _normalize_language(
            language_hint, self.settings.default_language_code
        )
        if language_code == "unknown":
            from_name = self.handwriting_language.detect_from_name(path)
            if from_name["language_code"] != "unknown":
                language_code = str(from_name["language_code"])
                language_family = str(from_name["language_family"])
                language_confidence = float(from_name["confidence"])
                source_hints["language_source"] = from_name["source"]
            elif document_type in {DocumentType.HANDWRITTEN, DocumentType.MIXED} and page_paths:
                detected = self.handwriting_language.detect_from_pages(page_paths)
                language_code = str(detected["language_code"])
                language_family = str(detected["language_family"])
                language_confidence = float(detected["confidence"])
                source_hints["language_source"] = detected["source"]
                if "scores" in detected:
                    source_hints["language_scores"] = detected["scores"]
            else:
                warnings.append(
                    "No language hint supplied. Routing will stay conservative and may default to your requested model family only."
                )
                source_hints["language_source"] = "unknown"

        if page_paths:
            try:
                with Image.open(page_paths[0]) as image:
                    width, height = image.size
                source_hints["image_size"] = {"width": width, "height": height}
                if width > 2200 or height > 2200:
                    layout_complexity = LayoutComplexity.HIGH
                    needs_layout_model = True
            except Exception:
                warnings.append("Image size inspection failed.")

        if len(prepared_document.pages) > 1:
            source_hints["page_count"] = len(prepared_document.pages)
            needs_layout_model = True
            if layout_complexity == LayoutComplexity.LOW:
                layout_complexity = LayoutComplexity.MEDIUM

        return AnalysisResult(
            document_type=document_type,
            handwritten_confidence=handwritten_confidence,
            language_code=language_code,
            language_family=language_family,
            language_confidence=language_confidence,
            layout_complexity=layout_complexity,
            needs_table_model=needs_table_model,
            needs_formula_model=needs_formula_model,
            needs_layout_model=needs_layout_model,
            source_hints=source_hints,
            warnings=warnings,
        )
