# ═══════════════════════════════════════════════════════════════════
#  MedAI Context Graph Service — Neo4j Patient Memory
# ═══════════════════════════════════════════════════════════════════

FROM python:3.11-slim AS runtime

ARG BUILD_DATE
LABEL maintainer="MedAI Team" \
      org.opencontainers.image.title="MedAI Context Graph Service"

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml /app/
RUN pip install --upgrade pip && pip install .

COPY . /app/

EXPOSE 8007

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -fsSL http://localhost:8007/health || exit 1

CMD ["python", "-m", "uvicorn", "context_graph_service.app:app", \
     "--host", "0.0.0.0", "--port", "8007", "--workers", "1"]
