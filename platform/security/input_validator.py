"""
Input validation — file uploads, text sanitization, injection prevention.

Production features:
    - Comprehensive magic-byte validation (PDF, JPEG, PNG, TIFF, DICOM, NIfTI, MP4)
    - Extension vs content-type mismatch detection
    - Executable blocklist (double extension attacks)
    - SQL injection pattern detection
    - Prompt injection detection (for AI queries)
    - Virus scan hook (ClamAV integration)
    - Content-length enforcement
    - Unicode normalization for consistent comparison
"""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from pathlib import Path, PurePosixPath
from typing import Any

from fastapi import HTTPException, UploadFile, status

from config import settings

logger = logging.getLogger(__name__)


# ── Dangerous Patterns ───────────────────────────────────────────────────

_XSS_PATTERNS = re.compile(
    r"<script|</script|javascript:|vbscript:|data:text/html|"
    r"on\w+\s*=|eval\s*\(|exec\s*\(|Function\s*\(|"
    r"__import__|subprocess|import\s+os|expression\s*\(|"
    r"document\.\w+|window\.\w+|\.innerHTML|\.outerHTML",
    re.IGNORECASE,
)

_SQL_INJECTION_PATTERNS = re.compile(
    r"('\s*(OR|AND)\s+')|"
    r"(;\s*(DROP|DELETE|UPDATE|INSERT|ALTER|EXEC)\s)|"
    r"(UNION\s+(ALL\s+)?SELECT)|"
    r"(--\s)|"
    r"(/\*.*\*/)|"
    r"(xp_\w+)|"
    r"(0x[0-9a-fA-F]+)",
    re.IGNORECASE,
)

_PROMPT_INJECTION_PATTERNS = re.compile(
    r"ignore\s+(all\s+)?previous\s+instructions|"
    r"you\s+are\s+now\s+|"
    r"pretend\s+to\s+be\s+|"
    r"system\s*:\s*|"
    r"\[INST\]|\[/INST\]|"
    r"<\|system\|>|<\|user\|>|<\|assistant\|>|"
    r"###\s*(system|instruction|human|assistant)",
    re.IGNORECASE,
)

# Null bytes and path traversal sequences
_PATH_TRAVERSAL = re.compile(r"\x00|\.\.[\\/]|%2e%2e|%00", re.IGNORECASE)

_EXECUTABLE_EXTENSIONS = {
    ".exe", ".bat", ".cmd", ".ps1", ".sh", ".com", ".msi",
    ".vbs", ".js", ".wsf", ".scr", ".pif", ".dll", ".sys",
    ".jar", ".py", ".rb", ".php", ".cgi", ".asp", ".aspx",
    ".jsp", ".war", ".elf", ".bin", ".app", ".dmg",
}

_ALLOWED_EXTENSIONS = {
    ".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".tif",
    ".dcm", ".nii", ".nii.gz",  # DICOM / NIfTI
    ".mp4", ".avi", ".mov",      # video
    ".bmp", ".webp",             # additional image formats
}

# ── Magic Bytes Database ─────────────────────────────────────────────────

_MAGIC_BYTES: dict[str, list[tuple[bytes, int]]] = {
    # Extension: [(magic_bytes, offset), ...]
    ".pdf":  [(b"%PDF", 0)],
    ".jpg":  [(b"\xff\xd8\xff", 0)],
    ".jpeg": [(b"\xff\xd8\xff", 0)],
    ".png":  [(b"\x89PNG\r\n\x1a\n", 0)],
    ".tiff": [(b"II\x2a\x00", 0), (b"MM\x00\x2a", 0)],  # Little-endian / Big-endian
    ".tif":  [(b"II\x2a\x00", 0), (b"MM\x00\x2a", 0)],
    ".dcm":  [(b"DICM", 128)],  # DICOM preamble at offset 128
    ".bmp":  [(b"BM", 0)],
    ".webp": [(b"RIFF", 0)],   # RIFF container (need WEBP at +8)
    ".mp4":  [(b"ftyp", 4)],   # ISO Base Media (offset 4)
    ".avi":  [(b"RIFF", 0)],   # RIFF container
    ".mov":  [(b"ftyp", 4), (b"moov", 4)],  # QuickTime
}

# NIfTI: magic at offset 344 = "n+1\0" or "ni1\0"
_NIFTI_MAGIC = [(b"n+1\x00", 344), (b"ni1\x00", 344)]


class FileValidator:
    """Validate uploaded files for safety and integrity."""

    @staticmethod
    async def validate(file: UploadFile) -> tuple[bytes, str]:
        """
        Validate and read an uploaded file.
        Returns (file_bytes, sha256_hex).
        Raises HTTPException on rejection.
        """
        # 1. Check filename exists
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No filename provided",
            )

        # 2. Sanitize filename — strip path components and null bytes
        safe_name = PurePosixPath(file.filename).name
        if _PATH_TRAVERSAL.search(safe_name) or "\x00" in safe_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid filename — path traversal detected",
            )

        # 3. Handle compound extensions (.nii.gz)
        ext = _get_extension(safe_name)

        # 4. Block executables (including double extension attacks)
        all_suffixes = Path(safe_name).suffixes
        for suffix in all_suffixes:
            if suffix.lower() in _EXECUTABLE_EXTENSIONS:
                logger.warning(
                    "Blocked executable upload: filename=%s ext=%s",
                    safe_name, suffix,
                )
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Executable files not allowed: {suffix}",
                )

        # 5. Allowlist check
        if ext not in _ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"File type not allowed: {ext}. "
                    f"Allowed: {', '.join(sorted(_ALLOWED_EXTENSIONS))}"
                ),
            )

        # 6. Read and check size
        content = await file.read()
        max_bytes = settings.max_upload_size_bytes
        if len(content) > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large ({len(content)} bytes). Max: {settings.max_upload_size_mb}MB",
            )

        # 7. Empty file check
        if len(content) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty file uploaded",
            )

        # 8. Magic-byte validation
        _validate_magic_bytes(content, ext, safe_name)

        # 9. Content-type vs extension mismatch
        _check_content_type_mismatch(file.content_type, ext)

        # 10. Virus scan hook (if enabled)
        if settings.enable_virus_scan:
            await _virus_scan(content, safe_name)

        # 11. Compute checksum
        sha256 = hashlib.sha256(content).hexdigest()

        logger.info(
            "File validated: name=%s size=%d ext=%s sha256=%s",
            safe_name, len(content), ext, sha256[:16] + "...",
        )

        return content, sha256


# ── Magic Byte Validation ────────────────────────────────────────────────

def _validate_magic_bytes(content: bytes, ext: str, filename: str) -> None:
    """Reject files whose magic bytes don't match their extension."""
    # NIfTI special case
    if ext in (".nii", ".nii.gz"):
        _check_nifti(content, filename)
        return

    signatures = _MAGIC_BYTES.get(ext)
    if not signatures:
        return  # No magic bytes to check for this extension

    for magic, offset in signatures:
        if len(content) > offset and content[offset:offset + len(magic)] == magic:
            return  # Match found

    # No signature matched
    logger.warning(
        "Magic byte mismatch: filename=%s expected_ext=%s",
        filename, ext,
    )
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"File content does not match {ext} format (magic byte check failed)",
    )


def _check_nifti(content: bytes, filename: str) -> None:
    """Validate NIfTI file (magic at offset 344)."""
    for magic, offset in _NIFTI_MAGIC:
        if len(content) > offset + len(magic):
            if content[offset:offset + len(magic)] == magic:
                return
    # NIfTI files can be gzipped — decompress header and validate inner magic
    if content[:2] == b"\x1f\x8b":
        import gzip
        try:
            # Only decompress enough to check the NIfTI header (first 348 bytes)
            with gzip.GzipFile(fileobj=__import__("io").BytesIO(content)) as gz:
                header = gz.read(348)
            for magic, offset in _NIFTI_MAGIC:
                if len(header) > offset + len(magic):
                    if header[offset:offset + len(magic)] == magic:
                        return
        except (gzip.BadGzipFile, OSError):
            pass  # Fall through to rejection
    logger.warning("NIfTI validation failed: filename=%s", filename)
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="File does not appear to be a valid NIfTI format",
    )


# ── Content-Type Mismatch ────────────────────────────────────────────────

_EXTENSION_TO_MIMES: dict[str, set[str]] = {
    ".pdf": {"application/pdf"},
    ".jpg": {"image/jpeg"},
    ".jpeg": {"image/jpeg"},
    ".png": {"image/png"},
    ".tiff": {"image/tiff"},
    ".tif": {"image/tiff"},
    ".dcm": {"application/dicom", "application/octet-stream"},
    ".mp4": {"video/mp4"},
    ".avi": {"video/x-msvideo", "video/avi"},
    ".mov": {"video/quicktime"},
    ".bmp": {"image/bmp"},
    ".webp": {"image/webp"},
}


def _check_content_type_mismatch(content_type: str | None, ext: str) -> None:
    """Warn on content-type vs extension mismatch (informational)."""
    if not content_type:
        return
    expected = _EXTENSION_TO_MIMES.get(ext)
    if expected and content_type.lower() not in expected:
        # Don't block, just warn — content-type can be spoofed by browsers
        logger.warning(
            "Content-type mismatch: declared=%s expected=%s for ext=%s",
            content_type, expected, ext,
        )


# ── Virus Scan Hook ──────────────────────────────────────────────────────

async def _virus_scan(content: bytes, filename: str) -> None:
    """
    ClamAV virus scan integration.

    Requires clamd running (via docker or system service).
    If ClamAV is unavailable, logs a warning and allows the file.
    """
    try:
        import clamd
        scanner = clamd.ClamdUnixSocket()
        result = scanner.instream(content)
        status_result = result.get("stream", ("OK", None))
        if status_result[0] != "OK":
            logger.warning(
                "Virus detected in upload: filename=%s virus=%s",
                filename, status_result[1],
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File rejected: malware detected",
            )
    except ImportError:
        logger.debug("ClamAV (clamd) not installed — skipping virus scan")
    except ConnectionError as exc:
        # ClamAV daemon not reachable — fail-closed in production
        logger.error("Virus scan unavailable (connection failed): %s", exc)
        if getattr(settings, "virus_scan_fail_closed", True):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Virus scan service unavailable. Upload rejected for safety.",
            )
    except Exception as exc:
        logger.error("Virus scan error: %s", exc)
        if getattr(settings, "virus_scan_fail_closed", True):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Virus scan failed. Upload rejected for safety.",
            )


# ── Text Sanitization ────────────────────────────────────────────────────

def sanitize_text(text: str, max_length: int = 10_000) -> str:
    """
    Sanitize user text input for safety.

    Defenses:
        1. Unicode normalization (NFKC)
        2. Null byte removal
        3. XSS pattern stripping
        4. SQL injection pattern stripping
        5. Length limiting
    """
    if not text:
        return ""

    # Unicode normalization (prevents homoglyph attacks)
    text = unicodedata.normalize("NFKC", text)

    # Remove null bytes
    text = text.replace("\x00", "")

    # Strip XSS patterns
    text = _XSS_PATTERNS.sub("[removed]", text)

    # Strip SQL injection patterns
    if _SQL_INJECTION_PATTERNS.search(text):
        logger.warning("SQL injection pattern detected and sanitized")
        text = _SQL_INJECTION_PATTERNS.sub("[removed]", text)

    # Limit length
    return text[:max_length]


def sanitize_ai_query(query: str, max_length: int = 10_000) -> str:
    """
    Sanitize text intended for AI model input.

    Additional check: prompt injection patterns.
    """
    clean = sanitize_text(query, max_length)

    # Detect prompt injection attempts
    if _PROMPT_INJECTION_PATTERNS.search(clean):
        logger.warning("Prompt injection attempt detected")
        clean = _PROMPT_INJECTION_PATTERNS.sub("[removed]", clean)

    return clean


def sanitize_search_query(query: str) -> str:
    """
    Sanitize search queries — strip SQL wildcards for LIKE operations.

    Prevents wildcard injection in database LIKE queries.
    """
    clean = sanitize_text(query, max_length=500)
    # Escape SQL LIKE wildcards
    clean = clean.replace("%", "\\%").replace("_", "\\_")
    return clean


# ── Helpers ──────────────────────────────────────────────────────────────

def _get_extension(filename: str) -> str:
    """Get file extension, handling compound extensions like .nii.gz."""
    name = filename.lower()
    if name.endswith(".nii.gz"):
        return ".nii.gz"
    return Path(name).suffix
