# ═══════════════════════════════════════════════════════════════════
#  IBM Granite 4.0 3B Vision — vLLM Sidecar
#  OpenAI-compatible serving for document extraction
#
#  Build:  docker build -t granite-vision-sidecar .
#  Run:    docker run --gpus 1 -p 8005:8005 granite-vision-sidecar
# ═══════════════════════════════════════════════════════════════════

FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04 AS runtime

ARG BUILD_DATE
LABEL maintainer="MedAI Team" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.title="Granite Vision 4.0 3B Sidecar" \
      org.opencontainers.image.description="vLLM serving for IBM Granite 4.0 3B Vision document extraction"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip curl && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install vLLM + dependencies
RUN pip install --upgrade pip && \
    pip install \
        vllm \
        torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128 && \
    pip install \
        transformers==4.57.6 \
        peft==0.18.1 \
        tokenizers==0.22.2 \
        pillow==12.1.1 \
        huggingface-hub

# Copy custom Granite serving scripts (downloaded at build or mounted)
COPY start_server.py /app/start_server.py

EXPOSE 8005

HEALTHCHECK --interval=30s --timeout=15s --start-period=300s --retries=5 \
    CMD curl -fsSL http://localhost:8005/health || exit 1

# Default: full-merge mode (fastest inference)
CMD ["python", "start_server.py", \
     "--model", "ibm-granite/granite-4.0-3b-vision", \
     "--trust_remote_code", \
     "--host", "0.0.0.0", \
     "--port", "8005", \
     "--hf-overrides", "{\"adapter_path\": \"ibm-granite/granite-4.0-3b-vision\"}"]
