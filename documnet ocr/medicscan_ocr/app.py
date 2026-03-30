from __future__ import annotations

import logging
import os
import tempfile
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from medicscan_ocr.config import load_settings
from medicscan_ocr.service import MediScanOCRService
from medicscan_ocr.utils.files import SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)


class _AppState:
    settings = None
    service = None


_state = _AppState()


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid integer env var %s=%r; using default %s", name, raw, default)
        return default


@asynccontextmanager
async def lifespan(app: FastAPI):
    if _state.settings is None:
        _state.settings = load_settings()
    if _state.service is None:
        _state.service = MediScanOCRService(_state.settings)
    yield


app = FastAPI(
    title="MediScan OCR-X",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

ALLOWED_ORIGINS = os.getenv("MEDISCAN_CORS_ORIGINS", "").split(",")
ALLOWED_ORIGINS = [origin.strip() for origin in ALLOWED_ORIGINS if origin.strip()]

if not ALLOWED_ORIGINS:
    logger.warning(
        "MEDISCAN_CORS_ORIGINS not set — using local development origins only. "
        "Set MEDISCAN_CORS_ORIGINS for production (e.g. 'https://app.example.com')."
    )
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS or ["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key"],
)

MAX_UPLOAD_SIZE_BYTES = _env_int("MEDISCAN_MAX_UPLOAD_BYTES", 100 * 1024 * 1024)


@app.get("/health")
def health():
    backend_status = _state.service.registry.all_statuses()
    ready_backends = [name for name, info in backend_status.items() if info.get("mode") == "ready"]
    return {
        "status": "ok" if ready_backends else "degraded",
        "sarvam_key_configured": _state.service.settings.has_sarvam_key,
        "backends": backend_status,
        "ready_backends": ready_backends,
        "preprocessing_enabled": _state.service.settings.enable_preprocessing,
    }


@app.post("/ocr")
async def ocr(
    file: UploadFile = File(...),
    language_hint: Optional[str] = Form(None),
    backend: str = Form("auto"),
    document_type_hint: Optional[str] = Form(None),
    dry_run: bool = Form(False),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type '{0}'. Supported: {1}".format(
                suffix, ", ".join(sorted(SUPPORTED_EXTENSIONS))
            ),
        )

    tmp_dir = Path(tempfile.mkdtemp(prefix="mediscan_upload_"))
    try:
        safe_name = Path(file.filename).name
        tmp_file = tmp_dir / safe_name
        size = 0
        with open(tmp_file, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_SIZE_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="File exceeds maximum upload size of {0} MB.".format(
                            MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)
                        ),
                    )
                f.write(chunk)

        logger.info("Processing uploaded file: %s (%d bytes)", safe_name, size)

        result = _state.service.process(
            input_path=str(tmp_file),
            language_hint=language_hint,
            backend=backend,
            document_type_hint=document_type_hint,
            dry_run=dry_run,
        )
        return result.to_dict()
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        logger.error("File not found during processing: %s", exc)
        raise HTTPException(status_code=404, detail="Requested file not found.")
    except ValueError as exc:
        logger.error("Validation error: %s", exc)
        raise HTTPException(status_code=400, detail="Invalid input or processing error.")
    except Exception as exc:
        logger.exception("Unexpected error during OCR processing")
        raise HTTPException(status_code=500, detail="Internal processing error.")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
