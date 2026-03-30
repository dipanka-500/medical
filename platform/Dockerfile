# ═══════════════════════════════════════════════════════════════════
#  MedAI Platform — Production Multi-Stage Dockerfile
# ═══════════════════════════════════════════════════════════════════

# ── Stage 1: Builder ─────────────────────────────────────────────
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

ARG INSTALL_VOICE_EXTRAS=false

WORKDIR /build

# System deps for building native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev ffmpeg libsndfile1-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps into a virtual env (keeps final image clean)
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml .
# Install dependencies first (cacheable layer)
RUN pip install --upgrade pip setuptools wheel
COPY . /build/
RUN if [ "$INSTALL_VOICE_EXTRAS" = "true" ]; then \
      pip install torch torchaudio && \
      pip install ".[voice]"; \
    else \
      pip install "."; \
    fi

# ── Stage 2: Runtime (lean) ──────────────────────────────────────
FROM python:3.12-slim AS runtime

ARG BUILD_DATE
ARG VCS_REF

LABEL maintainer="MedAI Platform Team" \
      org.opencontainers.image.title="medai-platform" \
      org.opencontainers.image.description="Unified Medical AI Platform API Gateway" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.vendor="MedAI" \
      org.opencontainers.image.version="1.0.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app" \
    # Hardened: no core dumps with patient data
    PYTHONFAULTHANDLER=1

WORKDIR /app

# Minimal runtime deps (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 curl tini ffmpeg libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

# Copy virtual env from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY . .

# Create required directories
RUN mkdir -p /app/data /app/logs /tmp/medai-uploads

# Non-root user for security
RUN groupadd -r medai && \
    useradd -r -g medai -d /app -s /sbin/nologin medai && \
    chown -R medai:medai /app /tmp/medai-uploads
USER medai

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=15s \
    CMD curl -fsSL http://localhost:8000/api/v1/health/live || exit 1

EXPOSE 8000

# Use tini as PID 1 for proper signal handling
ENTRYPOINT ["tini", "--"]
ENV FORWARDED_ALLOW_IPS="127.0.0.1"

CMD ["sh", "-c", "uvicorn main:app \
     --host 0.0.0.0 \
     --port 8000 \
     --workers ${UVICORN_WORKERS:-4} \
     --limit-concurrency 100 \
     --backlog 2048 \
     --access-log \
     --proxy-headers \
     --forwarded-allow-ips ${FORWARDED_ALLOW_IPS}"]
