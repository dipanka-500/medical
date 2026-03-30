from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Iterable, List


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".zip",
    ".doc",
    ".docx",
    ".bmp",
    ".tif",
    ".tiff",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def ensure_supported_file(path: str | Path) -> Path:
    file_path = Path(path).expanduser().resolve()
    if not file_path.exists():
        raise FileNotFoundError("Input file does not exist: {0}".format(file_path))
    if file_path.is_dir():
        raise IsADirectoryError("Expected a file path, got a directory: {0}".format(file_path))
    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            "Unsupported file type {0}. Supported: {1}".format(
                file_path.suffix.lower(), ", ".join(sorted(SUPPORTED_EXTENSIONS))
            )
        )
    return file_path


def is_image_path(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def is_pdf_path(path: str | Path) -> bool:
    return Path(path).suffix.lower() == ".pdf"


def is_office_document_path(path: str | Path) -> bool:
    return Path(path).suffix.lower() in {".doc", ".docx"}


def guess_media_type(path: str | Path) -> str:
    media_type, _ = mimetypes.guess_type(str(path))
    return media_type or "application/octet-stream"


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def flatten_file_list(paths: Iterable[str | Path]) -> List[Path]:
    return [ensure_supported_file(path) for path in paths]
