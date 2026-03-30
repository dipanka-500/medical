# ═══════════════════════════════════════════════════════════════════
#  MediScan OCR Engine — Production Dockerfile
# ═══════════════════════════════════════════════════════════════════

# ── Stage 1: Builder ─────────────────────────────────────────────
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

# System deps for OpenCV and building native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml .
# Install dependencies first (cacheable layer)
RUN pip install --upgrade pip setuptools wheel
COPY . /build/
# Keep local/default builds lean and only opt into heavyweight extras when needed.
ARG ENABLE_GPU=false
ARG ENABLE_SARVAM=false
RUN extras="docsplit"; \
    if [ "$ENABLE_SARVAM" = "true" ]; then extras="${extras},sarvam"; fi; \
    if [ "$ENABLE_GPU" = "true" ]; then extras="${extras},gpu"; fi; \
    pip install ".[${extras}]"

# ── Stage 2: Runtime ─────────────────────────────────────────────
FROM python:3.12-slim AS runtime

ARG BUILD_DATE

LABEL maintainer="MediScan OCR Team" \
      org.opencontainers.image.title="mediscan-ocr" \
      org.opencontainers.image.description="MediScan OCR-X Adaptive Pipeline" \
      org.opencontainers.image.created="${BUILD_DATE}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app"

WORKDIR /app

# Minimal runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 curl tini && \
    rm -rf /var/lib/apt/lists/*

# Copy virtual env from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY . .

# Create required directories
RUN mkdir -p /app/outputs /tmp/mediscan-ocr-uploads

# Non-root user
RUN groupadd -r mediscan && \
    useradd -r -g mediscan -d /app -s /sbin/nologin mediscan && \
    chown -R mediscan:mediscan /app /tmp/mediscan-ocr-uploads
USER mediscan

ENV PORT=8003
EXPOSE ${PORT}

HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=15s \
    CMD curl -fsSL http://localhost:${PORT}/health || exit 1

ENTRYPOINT ["tini", "--"]
CMD ["sh", "-c", "uvicorn medicscan_ocr.app:app --host 0.0.0.0 --port ${PORT:-8003} --workers 2"]
