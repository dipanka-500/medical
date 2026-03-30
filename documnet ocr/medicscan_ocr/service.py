from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from medicscan_ocr.analysis import DocumentIntelligenceAnalyzer
from medicscan_ocr.backends import build_registry
from medicscan_ocr.config import Settings
from medicscan_ocr.document import DocumentPreparer
from medicscan_ocr.fusion import fuse_backend_results
from medicscan_ocr.routing import RoutePlanner
from medicscan_ocr.schemas import BackendResult, OCRResult, RoutingDecision
from medicscan_ocr.utils.files import ensure_directory, ensure_supported_file

logger = logging.getLogger(__name__)


class MediScanOCRService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.preparer = DocumentPreparer(
            artifacts_dir=settings.artifacts_dir,
            enable_preprocessing=settings.enable_preprocessing,
        )
        self.analyzer = DocumentIntelligenceAnalyzer(settings)
        self.route_planner = RoutePlanner(settings)
        self.registry = build_registry(settings)

    def process(
        self,
        input_path: str,
        language_hint: Optional[str] = None,
        backend: str = "auto",
        document_type_hint: Optional[str] = None,
        dry_run: bool = False,
    ) -> OCRResult:
        source = ensure_supported_file(input_path)
        logger.info("Processing: %s (backend=%s, dry_run=%s)", source.name, backend, dry_run)

        prepared_document = self.preparer.prepare(str(source))
        logger.info(
            "Prepared: type=%s, pages=%d",
            prepared_document.source_type, len(prepared_document.pages),
        )

        analysis = self.analyzer.analyze(
            prepared_document=prepared_document,
            language_hint=language_hint,
            document_type_hint=document_type_hint,
        )
        logger.info(
            "Analysis: doc_type=%s, language=%s (%s), hw_conf=%.2f",
            analysis.document_type.value, analysis.language_code,
            analysis.language_family, analysis.handwritten_confidence,
        )

        route = self.route_planner.decide(analysis=analysis, requested_backend=backend)
        route = self._filter_unavailable_backends(route)
        logger.info(
            "Route: primary=%s, secondaries=%s",
            route.primary_backend, route.secondary_backends,
        )

        backend_results = []
        planned = [route.primary_backend] + route.secondary_backends
        if dry_run:
            from medicscan_ocr.schemas import BackendResult

            backend_results = [
                BackendResult(
                    backend=name,
                    status="skipped",
                    metadata={
                        "reason": "dry_run",
                        "backend_input_path": prepared_document.backend_input_path
                        if name != "sarvam_vision"
                        else prepared_document.native_path,
                    },
                )
                for name in planned
            ]
            return fuse_backend_results(
                input_path=str(source),
                preprocessed_path=prepared_document.pages[0].processed_path
                if prepared_document.pages
                else prepared_document.native_path,
                analysis=analysis,
                route=route,
                backend_results=backend_results,
            )

        for name in planned:
            if not self.registry.has(name):
                backend_results.append(
                    BackendResult(
                        backend=name,
                        status="failed",
                        error="Backend not registered: {0}".format(name),
                    )
                )
                continue
            available, detail = self.registry.availability(name)
            if not available:
                status = "skipped" if detail == "placeholder" else "failed"
                backend_results.append(
                    BackendResult(
                        backend=name,
                        status=status,
                        error=None if status == "skipped" else detail,
                        metadata={"availability": detail},
                    )
                )
                continue
            backend_impl = self.registry.get(name)
            target_dir = ensure_directory(self.settings.artifacts_dir / name / source.stem)
            backend_input = (
                prepared_document.native_path
                if name == "sarvam_vision"
                else prepared_document.backend_input_path
            )
            logger.info("Running backend: %s", name)
            result = backend_impl.run(
                input_path=backend_input,
                analysis=analysis,
                output_dir=str(target_dir),
            )
            if prepared_document.warnings:
                result.metadata.setdefault("input_warnings", prepared_document.warnings)
            backend_results.append(result)
            logger.info(
                "Backend %s: status=%s, confidence=%.2f",
                name, result.status, result.confidence,
            )
            if result.status == "completed":
                continue

        return fuse_backend_results(
            input_path=str(source),
            preprocessed_path=prepared_document.pages[0].processed_path
            if prepared_document.pages
            else prepared_document.native_path,
            analysis=analysis,
            route=route,
            backend_results=backend_results,
        )

    def write_result(self, result: OCRResult, output_path: Optional[str] = None) -> Path:
        target = Path(output_path) if output_path else self.settings.output_dir / "{0}.json".format(Path(result.input_path).stem)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return target

    def _filter_unavailable_backends(self, route: RoutingDecision) -> RoutingDecision:
        planned = [route.primary_backend, *route.secondary_backends]
        ready: list[str] = []
        removed: list[str] = []
        reasons = list(route.reason)

        for name in planned:
            if not self.registry.has(name):
                removed.append(f"{name} (not registered)")
                continue
            available, detail = self.registry.availability(name)
            if available:
                ready.append(name)
                continue
            removed.append(f"{name} ({detail or 'unavailable'})")

        if not ready:
            return route

        if removed:
            reasons.append(
                "Skipped unavailable OCR backends: {0}.".format(", ".join(removed))
            )

        return RoutingDecision(
            primary_backend=ready[0],
            secondary_backends=ready[1:],
            enrichers=route.enrichers,
            reason=reasons,
            requested_backend=route.requested_backend,
        )
