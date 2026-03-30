# ═══════════════════════════════════════════════════════════════════
#  Chroma Context-1 Agent — Multi-hop Retrieval (Stub Harness)
# ═══════════════════════════════════════════════════════════════════

FROM python:3.11-slim AS runtime

ARG BUILD_DATE
LABEL maintainer="MedAI Team" \
      org.opencontainers.image.title="MedAI Context-1 Agent"

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml /app/
RUN pip install --upgrade pip && pip install .

COPY . /app/

EXPOSE 8008

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -fsSL http://localhost:8008/health || exit 1

CMD ["python", "-m", "uvicorn", "context1_agent.app:app", \
     "--host", "0.0.0.0", "--port", "8008", "--workers", "1"]
