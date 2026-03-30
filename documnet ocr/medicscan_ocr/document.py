from __future__ import annotations

import importlib.util
import logging
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple

from medicscan_ocr.preprocess import preprocess_file

logger = logging.getLogger(__name__)
from medicscan_ocr.schemas import PreparedDocument, PreparedPage
from medicscan_ocr.utils.archive import safe_extract_zip
from medicscan_ocr.utils.files import (
    ensure_directory,
    ensure_supported_file,
    is_image_path,
    is_office_document_path,
    is_pdf_path,
)
from medicscan_ocr.utils.sorting import natural_sorted_paths


class DocumentPreparer:
    def __init__(self, artifacts_dir: str | Path, enable_preprocessing: bool = True) -> None:
        self.artifacts_dir = Path(artifacts_dir).resolve()
        self.enable_preprocessing = enable_preprocessing

    def prepare(self, input_path: str) -> PreparedDocument:
        source = ensure_supported_file(input_path)
        suffix = source.suffix.lower()
        warnings: List[str] = []

        if is_image_path(source):
            pages = self._prepare_image_pages([source])
            return PreparedDocument(
                input_path=str(source),
                source_type="image",
                native_path=str(source),
                backend_input_path=pages[0].processed_path,
                pages=pages,
            )

        if suffix == ".zip":
            extract_dir = ensure_directory(self.artifacts_dir / "bundles" / source.stem / "zip_input")
            safe_extract_zip(source, extract_dir)
            image_paths = natural_sorted_paths(
                [path for path in extract_dir.rglob("*") if path.is_file() and is_image_path(path)]
            )
            if not image_paths:
                raise ValueError("ZIP input does not contain any supported image files.")
            pages = self._prepare_image_pages(image_paths)
            backend_dir = str(Path(pages[0].processed_path).parent)
            return PreparedDocument(
                input_path=str(source),
                source_type="zip",
                native_path=str(source),
                backend_input_path=backend_dir,
                pages=pages,
                metadata={"extract_dir": str(extract_dir)},
            )

        if is_office_document_path(source):
            converted_pdf, conversion_warnings = self._convert_office_to_pdf(source)
            warnings.extend(conversion_warnings)
            if converted_pdf is None:
                return PreparedDocument(
                    input_path=str(source),
                    source_type="office_document",
                    native_path=str(source),
                    backend_input_path=str(source),
                    warnings=warnings,
                )
            source = converted_pdf
            suffix = source.suffix.lower()

        if is_pdf_path(source):
            rendered_pages, render_warnings = self._render_pdf_pages(source)
            warnings.extend(render_warnings)
            if rendered_pages:
                pages = self._prepare_image_pages(rendered_pages)
                backend_dir = str(Path(pages[0].processed_path).parent)
                return PreparedDocument(
                    input_path=str(Path(input_path).resolve()),
                    source_type="pdf",
                    native_path=str(source),
                    backend_input_path=backend_dir,
                    pages=pages,
                    warnings=warnings,
                )
            return PreparedDocument(
                input_path=str(Path(input_path).resolve()),
                source_type="pdf",
                native_path=str(source),
                backend_input_path=str(source),
                warnings=warnings,
            )

        return PreparedDocument(
            input_path=str(Path(input_path).resolve()),
            source_type="file",
            native_path=str(source),
            backend_input_path=str(source),
            warnings=warnings,
        )

    def _prepare_image_pages(self, image_paths: List[Path]) -> List[PreparedPage]:
        pages = []
        for index, image_path in enumerate(natural_sorted_paths(image_paths), start=1):
            preprocess_result = preprocess_file(
                input_path=str(image_path),
                artifacts_dir=str(self.artifacts_dir / "page_preprocessed"),
                enabled=self.enable_preprocessing,
            )
            pages.append(
                PreparedPage(
                    page_number=index,
                    source_name=image_path.name,
                    original_path=str(image_path.resolve()),
                    processed_path=preprocess_result.processed_path,
                    metadata=preprocess_result.metadata,
                )
            )
        return pages

    def _convert_office_to_pdf(self, source: Path) -> Tuple[Path | None, List[str]]:
        warnings: List[str] = []
        output_dir = ensure_directory(self.artifacts_dir / "office_converted" / source.stem)
        output_pdf = output_dir / "{0}.pdf".format(source.stem)

        soffice = shutil.which("soffice")
        if soffice:
            command = [
                soffice,
                "--headless",
                "--convert-to",
                "pdf",
                "--outdir",
                str(output_dir),
                str(source),
            ]
            result = subprocess.run(command, check=False, capture_output=True, text=True)
            if output_pdf.exists():
                return output_pdf, warnings
            warnings.append(result.stderr.strip() or "LibreOffice conversion failed.")

        if importlib.util.find_spec("docx2pdf"):
            import sys
            command = [sys.executable, "-m", "docx2pdf", str(source), str(output_pdf)]
            result = subprocess.run(command, check=False, capture_output=True, text=True)
            if output_pdf.exists():
                return output_pdf, warnings
            warnings.append(result.stderr.strip() or "docx2pdf conversion failed.")

        warnings.append(
            "Office documents need LibreOffice or docx2pdf for conversion before OCR page splitting."
        )
        return None, warnings

    def _render_pdf_pages(self, source: Path) -> Tuple[List[Path], List[str]]:
        warnings: List[str] = []
        output_dir = ensure_directory(self.artifacts_dir / "pdf_pages" / source.stem)

        if importlib.util.find_spec("pypdfium2"):
            import pypdfium2 as pdfium

            document = pdfium.PdfDocument(str(source))
            try:
                page_paths = []
                for index in range(len(document)):
                    page = document[index]
                    image = page.render(scale=2.0).to_pil().convert("RGB")
                    page_path = output_dir / "page_{0:04d}.png".format(index + 1)
                    image.save(page_path)
                    page_paths.append(page_path)
                return page_paths, warnings
            finally:
                document.close()

        warnings.append(
            "PDF page splitting requires pypdfium2 in this local build. Falling back to native PDF submission."
        )
        return [], warnings
