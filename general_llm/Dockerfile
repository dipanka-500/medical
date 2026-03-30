# ═══════════════════════════════════════════════════════════════════
#  General LLM Service — OpenAI-compatible local model inference
# ═══════════════════════════════════════════════════════════════════
#
# Serves Qwen2.5-7B-Instruct (or any HF model) with an OpenAI-compatible
# API. The platform gateway calls /v1/chat/completions just like OpenAI.
#
# Build:
#   docker build -t general-llm .
#
# Run (CPU):
#   docker run -p 8004:8004 -v ./models:/models -e HF_HOME=/models general-llm
#
# Run (GPU):
#   docker run --gpus all -p 8004:8004 -v ./models:/models -e HF_HOME=/models general-llm
#
# ═══════════════════════════════════════════════════════════════════

# ── Stage 1: Builder ─────────────────────────────────────────────
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml .
RUN pip install --upgrade pip setuptools wheel
COPY . /build/
RUN pip install "."

# ── Stage 2: Runtime ─────────────────────────────────────────────
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl tini && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY . .

# Non-root user
RUN groupadd -r llmuser && \
    useradd -r -g llmuser -d /app -s /sbin/nologin llmuser && \
    chown -R llmuser:llmuser /app
USER llmuser

ENV PORT=8004
EXPOSE ${PORT}

HEALTHCHECK --interval=30s --timeout=15s --retries=5 --start-period=300s \
    CMD curl -fsSL http://localhost:${PORT}/health || exit 1

ENTRYPOINT ["tini", "--"]
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8004} --workers 1 --timeout-keep-alive 120"]
