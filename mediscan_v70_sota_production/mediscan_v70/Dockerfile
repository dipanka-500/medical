FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

LABEL maintainer="MediScan AI Team"
LABEL version="7.0"
LABEL description="MediScan AI v7.0 — 16-Model Medical VLM Engine"

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Application code — install as a proper Python package so that
# relative imports (from .main import …) resolve correctly.
COPY . /app/mediscan_v70/

# Create necessary directories
RUN mkdir -p logs/audit data/rag

# Non-root user for security
RUN useradd -m -s /bin/bash mediscan && \
    chown -R mediscan:mediscan /app
USER mediscan

# Environment defaults
# SECURITY: API key must be set via runtime env var, not baked into image
# ENV MEDISCAN_API_KEY=""  # Removed — set at runtime via docker-compose or k8s
ENV MEDISCAN_CORS_ORIGINS="http://localhost:3000"
ENV MEDISCAN_PHI_SALT=""

ENV PORT=8001
EXPOSE ${PORT}

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD ["sh", "-c", "python -c \"import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')\""]

CMD ["sh", "-c", "uvicorn mediscan_v70.api_server:app --host 0.0.0.0 --port ${PORT:-8001} --workers 1"]
