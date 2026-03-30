"""
OpenRAG Service — FastAPI Application.

Production-ready agentic RAG with:
  - Document ingestion (Docling + dual indexing)
  - Hybrid search (vector + full-text + re-ranking)
  - Agentic multi-hop retrieval
  - Health checks and metrics
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from openrag_service.agentic_rag import AgenticRAGEngine
from openrag_service.config import load_config
from openrag_service.ingestion import IngestionPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


class _State:
    config = None
    pipeline: Optional[IngestionPipeline] = None
    agent: Optional[AgenticRAGEngine] = None


_state = _State()


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state.config = load_config()
    _state.pipeline = IngestionPipeline(_state.config)
    _state.agent = AgenticRAGEngine(_state.config, _state.pipeline)
    logger.info("OpenRAG service started on port %d", _state.config.port)
    yield
    if _state.agent:
        await _state.agent.close()


app = FastAPI(
    title="MedAI OpenRAG Service",
    version="0.1.0",
    description="Agentic RAG with Docling ingestion, hybrid search, and multi-hop retrieval",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

MAX_UPLOAD_BYTES = int(os.getenv("OPENRAG_MAX_UPLOAD_BYTES", str(100 * 1024 * 1024)))

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".csv", ".json", ".docx", ".doc", ".png", ".jpg", ".jpeg", ".tiff"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "openrag",
        "version": "0.1.0",
        "config": {
            "qdrant_url": _state.config.qdrant_url if _state.config else "",
            "opensearch_enabled": _state.config.opensearch_enabled if _state.config else False,
            "reranker_enabled": _state.config.reranker_enabled if _state.config else False,
            "docling_enabled": _state.config.enable_docling if _state.config else False,
        },
    }


@app.post("/ingest")
async def ingest_document(
    file: UploadFile = File(...),
    doc_type: str = Form("unknown"),
):
    """Ingest a document into the RAG pipeline."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}",
        )

    tmp_dir = Path(tempfile.mkdtemp(prefix="openrag_ingest_"))
    try:
        tmp_file = tmp_dir / Path(file.filename).name
        size = 0
        with open(tmp_file, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="File too large.")
                f.write(chunk)

        result = _state.pipeline.ingest(
            file_path=str(tmp_file),
            doc_type=doc_type,
            metadata={"original_filename": file.filename, "size_bytes": size},
        )
        return asdict(result)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/search")
async def search(
    query: str = Form(...),
    top_k: int = Form(10),
    use_reranker: bool = Form(True),
):
    """Hybrid search: vector + full-text with re-ranking."""
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    results = _state.pipeline.search(query, top_k=top_k, use_reranker=use_reranker)
    return {"query": query, "results": results, "count": len(results)}


@app.post("/query")
async def agentic_query(
    question: str = Form(...),
    top_k: int = Form(10),
):
    """Agentic multi-hop RAG query with decomposition and synthesis."""
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    result = await _state.agent.query(question, top_k=top_k)
    return asdict(result)


@app.post("/ingest/batch")
async def ingest_batch(
    files: list[UploadFile] = File(...),
    doc_type: str = Form("unknown"),
):
    """Batch ingest multiple documents."""
    results = []
    for file in files:
        tmp_dir = Path(tempfile.mkdtemp(prefix="openrag_batch_"))
        try:
            tmp_file = tmp_dir / Path(file.filename or "unknown").name
            content = await file.read()
            tmp_file.write_bytes(content)

            result = _state.pipeline.ingest(
                file_path=str(tmp_file),
                doc_type=doc_type,
                metadata={"original_filename": file.filename, "size_bytes": len(content)},
            )
            results.append(asdict(result))
        except Exception as exc:
            results.append({"filename": file.filename, "error": str(exc)})
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return {"ingested": len([r for r in results if "error" not in r]), "results": results}


if __name__ == "__main__":
    import uvicorn
    config = load_config()
    uvicorn.run(app, host=config.host, port=config.port)
