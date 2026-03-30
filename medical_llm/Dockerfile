# ═══════════════════════════════════════════════════════════════════
#  Medical LLM Engine — Production Multi-Stage Container
# ═══════════════════════════════════════════════════════════════════
#
# Build:
#   docker build -t medical-llm .
#
# Run (CPU):
#   docker run -p 8002:8002 medical-llm
#
# Run (GPU):
#   docker run --gpus all -p 8002:8002 medical-llm
#
# ═══════════════════════════════════════════════════════════════════

# ── Stage 1: Builder ─────────────────────────────────────────────
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

# System dependencies for faiss, spacy, and scientific python
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps into a virtual env
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml README.md ./
RUN pip install --upgrade pip setuptools wheel
COPY . /build/
ARG ENABLE_VLLM=false
RUN if [ "$ENABLE_VLLM" = "true" ]; then \
      pip install ".[vllm]"; \
    else \
      pip install "."; \
    fi

# ── Stage 2: Runtime ─────────────────────────────────────────────
FROM python:3.11-slim AS runtime

ARG BUILD_DATE

LABEL maintainer="MedAI Team" \
      org.opencontainers.image.title="medical-llm-engine" \
      org.opencontainers.image.description="Production Medical LLM Engine" \
      org.opencontainers.image.created="${BUILD_DATE}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1

WORKDIR /app

# Minimal runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl tini && \
    rm -rf /var/lib/apt/lists/*

# Copy virtual env from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY . .

# Create required directories
RUN mkdir -p /app/data/rag/faiss_index \
             /app/data/rag/pubmed_cache \
             /app/data/knowledge \
             /app/logs/audit

# Non-root user for security
RUN groupadd -r medllm && \
    useradd -r -g medllm -d /app -s /sbin/nologin medllm && \
    chown -R medllm:medllm /app
USER medllm

# Health check
ENV PORT=8002
EXPOSE ${PORT}

HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=120s \
    CMD curl -fsSL http://localhost:${PORT}/health || exit 1

# Use tini as PID 1 for signal handling
ENTRYPOINT ["tini", "--"]
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8002} --workers 1 --limit-concurrency 10 --backlog 128 --proxy-headers --access-log"]
