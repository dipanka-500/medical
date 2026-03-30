from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional

from medicscan_ocr.backends.base import OCRBackend
from medicscan_ocr.schemas import AnalysisResult, BackendResult, Section
from medicscan_ocr.utils.archive import safe_extract_zip
from medicscan_ocr.utils.text import extract_markdown_tables, normalize_text

logger = logging.getLogger(__name__)


class SarvamVisionBackend(OCRBackend):
    name = "sarvam_vision"

    def availability(self) -> tuple[bool, str | None]:
        if not self.settings.has_sarvam_key:
            return False, "SARVAM_API_KEY is not set"
        try:
            import sarvamai  # noqa: F401
        except ImportError:
            return False, "sarvamai package is not installed"
        return True, None

    def _build_client(self):
        if not self.settings.has_sarvam_key:
            raise RuntimeError("SARVAM_API_KEY is not set.")
        try:
            from sarvamai import SarvamAI
        except ImportError as exc:
            raise RuntimeError(
                "sarvamai is not installed. Run: pip install -e \".[sarvam]\""
            ) from exc
        return SarvamAI(api_subscription_key=self.settings.sarvam_api_key)

    def _language_for_job(self, analysis: AnalysisResult) -> str:
        if analysis.language_code != "unknown":
            return analysis.language_code
        return self.settings.default_language_code

    def _parse_zip(self, zip_path: Path, extract_dir: Path) -> BackendResult:
        extract_dir.mkdir(parents=True, exist_ok=True)
        safe_extract_zip(zip_path, extract_dir)

        markdown_parts = []
        html_parts = []
        sections: List[Section] = []
        extracted_files = []

        for path in sorted(extract_dir.rglob("*")):
            if not path.is_file():
                continue
            extracted_files.append(str(path))
            suffix = path.suffix.lower()
            if suffix in {".md", ".markdown"}:
                text = path.read_text(encoding="utf-8", errors="replace")
                markdown_parts.append(text)
                sections.append(Section(type="document", text=text, confidence=0.90))
            elif suffix == ".html":
                text = path.read_text(encoding="utf-8", errors="replace")
                html_parts.append(text)
                sections.append(Section(type="html", text=text, confidence=0.90))

        raw_text = normalize_text("\n\n".join(markdown_parts or html_parts))
        tables = extract_markdown_tables(raw_text)
        return BackendResult(
            backend=self.name,
            status="completed",
            raw_text=raw_text,
            sections=sections,
            confidence=0.90,
            metadata={
                "output_zip": str(zip_path),
                "extract_dir": str(extract_dir),
                "tables_detected": len(tables),
                "files": extracted_files,
            },
        )

    def run(
        self,
        input_path: str,
        analysis: AnalysisResult,
        output_dir: Optional[str] = None,
    ) -> BackendResult:
        try:
            client = self._build_client()
        except Exception as exc:
            return BackendResult(
                backend=self.name,
                status="failed",
                error=str(exc),
            )

        resolved_input = str(Path(input_path).resolve())
        language = self._language_for_job(analysis)
        base_output = Path(output_dir or self.settings.artifacts_dir / "sarvam").resolve()
        base_output.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Starting Sarvam job: input=%s, language=%s, format=%s",
            resolved_input, language, self.settings.sarvam_output_format,
        )

        try:
            job = client.document_intelligence.create_job(
                language=language,
                output_format=self.settings.sarvam_output_format,
            )
            job.upload_file(resolved_input)
            job.start()

            timeout = self.settings.sarvam_wait_timeout_seconds
            poll_interval = self.settings.sarvam_poll_interval_seconds
            start_time = time.monotonic()

            status = None
            while True:
                elapsed = time.monotonic() - start_time
                if elapsed > timeout:
                    return BackendResult(
                        backend=self.name,
                        status="failed",
                        error="Sarvam job timed out after {0:.0f} seconds.".format(timeout),
                        metadata={"job_id": getattr(job, "job_id", None)},
                    )
                try:
                    status = job.get_status()
                except Exception:
                    status = None

                job_state = ""
                if status is not None:
                    job_state = str(
                        getattr(status, "job_state", None)
                        or getattr(status, "state", "")
                    ).lower()

                if job_state in {"completed", "partiallycompleted"}:
                    break
                if job_state in {"failed", "error", "cancelled"}:
                    return BackendResult(
                        backend=self.name,
                        status="failed",
                        error="Sarvam job failed with state: {0}".format(job_state),
                        metadata={"job_id": getattr(job, "job_id", None)},
                    )

                # NOTE: This is a sync function — runs in a thread pool when called from
                # FastAPI. time.sleep is acceptable here as it blocks only the worker thread.
                time.sleep(poll_interval)

            job_id = getattr(job, "job_id", "sarvam_output")
            zip_path = base_output / "{0}.zip".format(job_id)
            job.download_output(str(zip_path))
            logger.info("Sarvam job %s completed, output saved to %s", job_id, zip_path)

            parsed = self._parse_zip(zip_path, base_output / zip_path.stem)
            metrics = {}
            try:
                metrics = job.get_page_metrics()
            except Exception:
                pass
            parsed.metadata["job_id"] = job_id
            parsed.metadata["language"] = language
            parsed.metadata["page_metrics"] = metrics
            parsed.metadata["job_state"] = job_state
            return parsed
        except FileNotFoundError:
            return BackendResult(
                backend=self.name,
                status="failed",
                error="Input file not found: {0}".format(resolved_input),
            )
        except Exception as exc:
            logger.exception("Sarvam backend failed")
            return BackendResult(
                backend=self.name,
                status="failed",
                error=str(exc),
            )
