#!/usr/bin/env bash
# ============================================================
# MediScan AI v7.0 — JarvisLabs Launch Script
# Instance: 8 × A6000 (PyTorch template)
#
# Usage:
#   bash deploy/jarvislabs/launch.sh             # full startup
#   bash deploy/jarvislabs/launch.sh --no-preload  # skip model pre-warm
#   bash deploy/jarvislabs/launch.sh --dry-run     # test without loading
#   bash deploy/jarvislabs/launch.sh --skip hulu_med_32b  # skip one model
#
# What this does:
#   1. Exports all env vars for 8-GPU operation
#   2. Optionally pre-loads all models into VRAM sequentially
#   3. Starts the FastAPI server on port 6006 (JarvisLabs public endpoint)
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"   # mediscan_v70_sota_production/
VENV_DIR="/home/venv"
LOG_DIR="$PROJECT_ROOT/logs"

mkdir -p "$LOG_DIR"

# ── Parse args ───────────────────────────────────────────────
NO_PRELOAD=0
DRY_RUN=""
SKIP_MODELS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-preload)   NO_PRELOAD=1 ;;
        --dry-run)      DRY_RUN="--dry-run" ;;
        --skip)         shift; SKIP_MODELS="$1" ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done

# ── Activate venv ─────────────────────────────────────────────
if [[ -d "$VENV_DIR" ]]; then
    source "$VENV_DIR/bin/activate"
else
    echo "[WARN] venv not found at $VENV_DIR — run setup.sh first"
fi

# ── Environment Variables (8 × A6000 tuning) ─────────────────

# Allow all 8 GPUs
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

# MediScan engine settings
export MEDISCAN_MAX_RESIDENT_MODELS=0          # 0 = keep ALL 16 models in VRAM (fits in 384 GB)
export MEDISCAN_AUTO_UNLOAD_AFTER_INFERENCE=0  # never evict — we have enough VRAM
export MEDISCAN_SEQUENTIAL_HEAVY_MODELS=1      # serialise LOADING of >20B models at startup

# HuggingFace — use /home which is on the 2TB persistent storage volume
# All model weights (~330 GB) are cached here; never re-downloaded after first run
export HF_HOME="/home/hf_cache"               # 2TB persistent disk → survives pause/resume
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_ENABLE_HF_TRANSFER=1            # faster downloads via hf_transfer
export TOKENIZERS_PARALLELISM=false           # avoid fork warnings

# PyTorch memory management
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,expandable_segments:True"
export CUDA_LAUNCH_BLOCKING=0                 # async GPU kernels

# Performance
export OMP_NUM_THREADS=8                      # match physical cores / GPU count
export MKL_NUM_THREADS=8

# Server (uses hardware_config.yaml but env vars override)
export MEDISCAN_HOST="0.0.0.0"
export MEDISCAN_PORT="6006"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  MediScan AI v7.0 — JarvisLabs 8×A6000 Launch           ║"
echo "╠══════════════════════════════════════════════════════════╣"
nvidia-smi --query-gpu=index,name,memory.total \
           --format=csv,noheader,nounits \
    | awk -F',' '{printf "║  GPU%d %-24s  %s MB VRAM          ║\n",$1,$2,$3}'
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ── HF cache dir ─────────────────────────────────────────────
mkdir -p "$HF_HOME"

# ── Optional: install hf_transfer for faster downloads ───────
pip show hf_transfer &>/dev/null \
    || pip install --quiet hf_transfer

# ── Pre-load models (sequential → parallel) ──────────────────
if [[ $NO_PRELOAD -eq 0 ]]; then
    echo "[INFO] Pre-loading all models into GPU VRAM…"
    echo "[INFO] This takes 10-30 min on first run (model downloads)."
    echo "[INFO] Subsequent runs use the cache and take ~3-5 min."
    echo ""

    SKIP_ARGS=""
    if [[ -n "$SKIP_MODELS" ]]; then
        SKIP_ARGS="--skip $SKIP_MODELS"
    fi

    python -m deploy.jarvislabs.sequential_loader \
        $DRY_RUN $SKIP_ARGS \
        2>&1 | tee "$LOG_DIR/preload_$(date +%Y%m%d_%H%M%S).log"

    echo "[INFO] Pre-load complete."
fi

# ── Start FastAPI server ──────────────────────────────────────
echo ""
echo "[INFO] Starting MediScan FastAPI server on 0.0.0.0:8080"
echo "[INFO] Access via the custom HTTP port 8080 you added in JarvisLabs dashboard"
echo "[INFO] Docs: http://<your-instance-url>/docs"
echo ""

cd "$PROJECT_ROOT"
exec python -m uvicorn mediscan_v70.api_server:app \
    --host 0.0.0.0 \
    --port 8080 \
    --workers 1 \
    --log-level info \
    --access-log \
    2>&1 | tee "$LOG_DIR/server_$(date +%Y%m%d_%H%M%S).log"
