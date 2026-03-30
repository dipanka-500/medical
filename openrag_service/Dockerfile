# ═══════════════════════════════════════════════════════════════════
#  MedAI OpenRAG Service — Agentic RAG with Docling
# ���══════════════════════════════════════════════════════════════════

FROM python:3.11-slim AS runtime

ARG BUILD_DATE
LABEL maintainer="MedAI Team" \
      org.opencontainers.image.title="MedAI OpenRAG Service" \
      org.opencontainers.image.created="${BUILD_DATE}"

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl libgl1 libglib2.0-0 poppler-utils tesseract-ocr && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml /app/
RUN pip install --upgrade pip && pip install ".[reranker]"

COPY . /app/

EXPOSE 8006

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -fsSL http://localhost:8006/health || exit 1

CMD ["python", "-m", "uvicorn", "openrag_service.app:app", \
     "--host", "0.0.0.0", "--port", "8006", "--workers", "1"]
