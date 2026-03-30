from __future__ import annotations

import json
import logging
import subprocess
import shutil
from pathlib import Path
from typing import List, Optional

from medicscan_ocr.backends.base import OCRBackend
from medicscan_ocr.schemas import AnalysisResult, BackendResult, Section
from medicscan_ocr.utils.sorting import natural_sort_key
from medicscan_ocr.utils.text import normalize_text

logger = logging.getLogger(__name__)


class LocalCommandBackend(OCRBackend):
    command_name = ""

    def availability(self) -> tuple[bool, str | None]:
        command = self.command_name
        if command == "surya_ocr" and shutil.which("surya"):
            return True, "surya command available"
        if shutil.which(command):
            return True, None
        return False, f"{command} command not found"

    def _run_command(self, command: List[str]) -> subprocess.CompletedProcess:
        logger.info("Running command: %s", " ".join(command))
        return subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=self.settings.local_command_timeout_seconds,
        )


class SuryaCommandBackend(LocalCommandBackend):
    name = "surya_command"
    command_name = "surya_ocr"

    def _build_command(self, input_path: str, target_dir: Path) -> List[str]:
        resolved = str(Path(input_path).resolve())
        # Surya v0.6+ uses `surya ocr` subcommand; older uses `surya_ocr`
        if shutil.which("surya_ocr"):
            return ["surya_ocr", resolved, "--output_dir", str(target_dir)]
        if shutil.which("surya"):
            return ["surya", "ocr", resolved, "--output_dir", str(target_dir)]
        return ["surya_ocr", resolved, "--output_dir", str(target_dir)]

    def _find_results_json(self, target_dir: Path) -> Optional[Path]:
        """Search for results.json - may be at root or nested in subdirectory."""
        direct = target_dir / "results.json"
        if direct.exists():
            return direct
        for candidate in sorted(target_dir.rglob("results.json")):
            return candidate
        for candidate in sorted(target_dir.rglob("*.json")):
            return candidate
        return None

    def run(
        self,
        input_path: str,
        analysis: AnalysisResult,
        output_dir: Optional[str] = None,
    ) -> BackendResult:
        target_dir = Path(output_dir or self.settings.artifacts_dir / "surya").resolve()
        target_dir.mkdir(parents=True, exist_ok=True)

        command = self._build_command(input_path, target_dir)

        try:
            result = self._run_command(command)
        except FileNotFoundError:
            return BackendResult(
                backend=self.name,
                status="failed",
                error="surya_ocr command not found. Install surya-ocr to enable this backend. "
                      "For v0.6+, the command is 'surya ocr'.",
            )
        except subprocess.TimeoutExpired:
            return BackendResult(
                backend=self.name,
                status="failed",
                error="surya_ocr timed out after {0} seconds.".format(
                    self.settings.local_command_timeout_seconds
                ),
            )

        if result.returncode != 0:
            return BackendResult(
                backend=self.name,
                status="failed",
                error=result.stderr.strip() or "surya_ocr failed with return code {0}".format(
                    result.returncode
                ),
                metadata={"stdout": result.stdout, "stderr": result.stderr},
            )

        results_path = self._find_results_json(target_dir)
        if results_path is None:
            return BackendResult(
                backend=self.name,
                status="failed",
                error="surya_ocr completed but no JSON output was found.",
                metadata={"stdout": result.stdout, "output_dir": str(target_dir)},
            )

        try:
            payload = json.loads(results_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            return BackendResult(
                backend=self.name,
                status="failed",
                error="Failed to parse surya_ocr output: {0}".format(exc),
            )

        page_sections = []
        lines = []
        confidences = []

        if isinstance(payload, dict):
            for _, pages in sorted(payload.items(), key=lambda item: natural_sort_key(item[0])):
                if not isinstance(pages, list):
                    pages = [pages]
                for page in pages:
                    text_lines = page.get("text_lines", [])
                    page_text = "\n".join(
                        item.get("text", "") for item in text_lines if item.get("text")
                    )
                    if page_text:
                        lines.append(page_text)
                        page_sections.append(
                            Section(
                                type="page",
                                text=page_text,
                                confidence=0.75,
                                data={"page_number": page.get("page")},
                            )
                        )
                    for item in text_lines:
                        confidence = item.get("confidence")
                        if isinstance(confidence, (float, int)):
                            confidences.append(float(confidence))
        elif isinstance(payload, list):
            for page_idx, page in enumerate(payload):
                text_lines = page.get("text_lines", [])
                page_text = "\n".join(
                    item.get("text", "") for item in text_lines if item.get("text")
                )
                if page_text:
                    lines.append(page_text)
                    page_sections.append(
                        Section(
                            type="page",
                            text=page_text,
                            confidence=0.75,
                            data={"page_number": page_idx + 1},
                        )
                    )
                for item in text_lines:
                    confidence = item.get("confidence")
                    if isinstance(confidence, (float, int)):
                        confidences.append(float(confidence))

        raw_text = normalize_text("\n\n".join(lines))
        confidence = sum(confidences) / len(confidences) if confidences else 0.75

        if not raw_text.strip():
            return BackendResult(
                backend=self.name,
                status="completed",
                raw_text="",
                sections=[],
                confidence=0.0,
                metadata={
                    "output_dir": str(target_dir),
                    "warning": "surya_ocr produced no text output",
                },
            )

        return BackendResult(
            backend=self.name,
            status="completed",
            raw_text=raw_text,
            sections=page_sections,
            confidence=round(confidence, 4),
            metadata={"output_dir": str(target_dir)},
        )


class ChandraCommandBackend(LocalCommandBackend):
    name = "chandra_command"
    command_name = "chandra"

    def run(
        self,
        input_path: str,
        analysis: AnalysisResult,
        output_dir: Optional[str] = None,
    ) -> BackendResult:
        target_dir = Path(output_dir or self.settings.artifacts_dir / "chandra").resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        command = [
            self.command_name,
            str(Path(input_path).resolve()),
            str(target_dir),
            "--method",
            "hf",
        ]
        try:
            result = self._run_command(command)
        except FileNotFoundError:
            return BackendResult(
                backend=self.name,
                status="failed",
                error="chandra command not found. Install chandra-ocr to enable this backend.",
            )
        except subprocess.TimeoutExpired:
            return BackendResult(
                backend=self.name,
                status="failed",
                error="chandra timed out after {0} seconds.".format(
                    self.settings.local_command_timeout_seconds
                ),
            )
        if result.returncode != 0:
            return BackendResult(
                backend=self.name,
                status="failed",
                error=result.stderr.strip() or "chandra failed with return code {0}".format(
                    result.returncode
                ),
                metadata={"stdout": result.stdout, "stderr": result.stderr},
            )

        markdown_paths = sorted(
            target_dir.rglob("*.md"), key=lambda path: natural_sort_key(path.as_posix())
        )
        if not markdown_paths:
            return BackendResult(
                backend=self.name,
                status="failed",
                error="chandra completed but markdown output was not found.",
                metadata={"output_dir": str(target_dir)},
            )

        chandra_metadata = {}
        metadata_paths = list(target_dir.rglob("*_metadata.json"))
        for meta_path in metadata_paths:
            try:
                chandra_metadata[meta_path.stem] = json.loads(
                    meta_path.read_text(encoding="utf-8")
                )
            except Exception:
                pass

        sections = []
        text_chunks = []
        for index, markdown_path in enumerate(markdown_paths, start=1):
            try:
                chunk = normalize_text(markdown_path.read_text(encoding="utf-8", errors="replace"))
            except Exception as exc:
                logger.warning("Failed to read chandra output %s: %s", markdown_path, exc)
                continue
            text_chunks.append(chunk)
            sections.append(
                Section(
                    type="page",
                    text=chunk,
                    confidence=0.82,
                    data={"page_number": index, "path": str(markdown_path)},
                )
            )
        raw_text = normalize_text("\n\n".join(text_chunks))
        return BackendResult(
            backend=self.name,
            status="completed",
            raw_text=raw_text,
            sections=sections,
            confidence=0.82,
            metadata={
                "output_dir": str(target_dir),
                "markdown_paths": [str(path) for path in markdown_paths],
                "chandra_metadata": chandra_metadata,
            },
        )
