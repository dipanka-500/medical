from __future__ import annotations

import logging
from typing import List

from medicscan_ocr.schemas import AnalysisResult, BackendResult, DocumentType, OCRResult, RoutingDecision, Section
from medicscan_ocr.utils.text import extract_markdown_tables, normalize_text, similarity_score

logger = logging.getLogger(__name__)


def fuse_backend_results(
    input_path: str,
    preprocessed_path: str,
    analysis: AnalysisResult,
    route: RoutingDecision,
    backend_results: List[BackendResult],
) -> OCRResult:
    completed = [result for result in backend_results if result.status == "completed"]
    best = None
    if completed:
        best = max(completed, key=lambda item: (item.confidence, len(item.raw_text)))
    else:
        best = backend_results[0] if backend_results else BackendResult(
            backend=route.primary_backend,
            status="failed",
            error="No backend results were produced.",
        )

    uncertain_regions = []
    if len(completed) >= 2:
        reference = completed[0].raw_text
        for candidate in completed[1:]:
            score = similarity_score(reference, candidate.raw_text)
            if score < 0.80:
                uncertain_regions.append(
                    "Low backend agreement: {0} vs {1} ({2:.2f})".format(
                        completed[0].backend, candidate.backend, score
                    )
                )

    raw_text = normalize_text(best.raw_text)
    structured = best.sections or [Section(type="document", text=raw_text, confidence=best.confidence)]
    tables = extract_markdown_tables(raw_text)

    return OCRResult(
        input_path=input_path,
        preprocessed_path=preprocessed_path,
        raw_text=raw_text,
        structured=structured,
        language=analysis.language_code,
        document_type=analysis.document_type.value,
        confidence=round(best.confidence, 4),
        tables=tables,
        handwritten_detected=analysis.document_type == DocumentType.HANDWRITTEN,
        uncertain_regions=uncertain_regions,
        route=route,
        analysis=analysis,
        backend_results=backend_results,
        metadata={
            "successful_backends": [item.backend for item in completed],
            "failed_backends": [item.backend for item in backend_results if item.status == "failed"],
            "skipped_backends": [item.backend for item in backend_results if item.status == "skipped"],
        },
    )
