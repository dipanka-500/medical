#!/usr/bin/env bash
# ============================================================
# MediScan AI v7.0 — JarvisLabs Setup Script
# Instance: PyTorch template, 8 × A6000 (384 GB VRAM)
#
# Run ONCE after creating your JarvisLabs instance:
#   bash deploy/jarvislabs/setup.sh
#
# What this does:
#   1. Verifies 8 A6000 GPUs are present
#   2. Installs uv (fast Python package manager)
#   3. Creates a persistent virtual environment in /home/venv
#      (survives instance pause/resume)
#   4. Installs all MediScan dependencies
#   5. Installs flash-attn (compiled for CUDA 12.x / A6000)
#   6. Symlinks the hardware config override into place
#   7. Sets up Jupyter kernel for interactive notebooks
# ============================================================
set -euo pipefail

VENV_DIR="/home/venv"
PROJECT_DIR="/home/$(whoami)/mediscan_v70_sota_production"
LOG_FILE="/home/setup_mediscan.log"

# ── Colours ─────────────────────────────────────────────────
GRN='\033[0;32m'; YLW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()    { echo -e "${GRN}[INFO]${NC}  $*" | tee -a "$LOG_FILE"; }
warn()    { echo -e "${YLW}[WARN]${NC}  $*" | tee -a "$LOG_FILE"; }
die()     { echo -e "${RED}[ERR] ${NC}  $*" | tee -a "$LOG_FILE"; exit 1; }

info "=== MediScan AI v7.0 JarvisLabs Setup ==="
info "Log: $LOG_FILE"

# ── 1. GPU check ─────────────────────────────────────────────
info "Checking GPUs…"
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
if [[ "$GPU_COUNT" -lt 8 ]]; then
    warn "Expected 8 A6000 GPUs, found $GPU_COUNT. Proceeding anyway."
else
    info "✓ $GPU_COUNT A6000 GPUs detected"
fi
nvidia-smi -L | tee -a "$LOG_FILE"

# ── 2. System packages ───────────────────────────────────────
info "Installing system packages…"
apt-get update -qq
apt-get install -y -qq \
    build-essential curl git wget \
    libgl1 libglib2.0-0 \
    dcmtk                    # DICOM CLI tools
info "✓ System packages installed"

# ── 3. Install uv ────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    info "Installing uv…"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi
info "✓ uv $(uv --version)"

# ── 4. Virtual environment in /home (persists after pause) ───
info "Creating virtual environment at $VENV_DIR…"
uv venv "$VENV_DIR" --python=python3.11 --seed
source "$VENV_DIR/bin/activate"
info "✓ venv active: $(python --version)"

# ── 5. PyTorch (CUDA 12.1 — matches JarvisLabs A6000 driver) ─
info "Installing PyTorch for CUDA 12.1…"
uv pip install --quiet \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print('  torch', torch.__version__, '| CUDA', torch.version.cuda)" \
    | tee -a "$LOG_FILE"

# ── 6. Flash Attention 2 (pre-built wheel for CUDA 12.x) ─────
info "Installing flash-attn 2 (pre-built)…"
# Use pre-built wheel to avoid 30+ min compilation
pip install --quiet flash-attn --no-build-isolation 2>>"$LOG_FILE" \
    || warn "flash-attn install failed — falling back to eager attention"

# ── 7. MediScan dependencies ─────────────────────────────────
info "Installing MediScan v7.0 dependencies…"

# Locate requirements.txt relative to this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ_FILE="$SCRIPT_DIR/../../requirements.txt"

if [[ -f "$REQ_FILE" ]]; then
    uv pip install --quiet -r "$REQ_FILE"
    info "✓ requirements.txt installed"
else
    warn "requirements.txt not found at $REQ_FILE — installing core packages only"
    uv pip install --quiet \
        transformers>=4.50.0 accelerate>=1.2.0 safetensors>=0.4.0 \
        open-clip-torch>=2.26.0 monai>=1.4.0 timm>=1.0.0 \
        pydicom nibabel SimpleITK Pillow opencv-python scikit-image \
        qwen-vl-utils>=0.0.8 \
        fastapi uvicorn[standard] python-multipart pydantic \
        chromadb sentence-transformers duckduckgo-search rank-bm25 \
        fhir.resources httpx structlog prometheus-client \
        cryptography python-jose[cryptography] \
        pyyaml numpy pandas tqdm rich aiofiles
fi

# ── 8. Jupyter kernel ─────────────────────────────────────────
info "Registering Jupyter kernel 'mediscan'…"
uv pip install --quiet ipykernel
python -m ipykernel install --user --name=mediscan --display-name "MediScan v7.0 (py3.11)"
info "✓ Kernel 'mediscan' registered — refresh JupyterLab and select it"

# ── 9. Activate in .bashrc (survives SSH reconnects) ─────────
BASHRC="$HOME/.bashrc"
ACTIVATE_LINE="source $VENV_DIR/bin/activate"
if ! grep -qF "$ACTIVATE_LINE" "$BASHRC"; then
    echo "" >> "$BASHRC"
    echo "# MediScan venv (auto-activated)" >> "$BASHRC"
    echo "$ACTIVATE_LINE" >> "$BASHRC"
    info "✓ Auto-activation added to ~/.bashrc"
fi

# ── 10. Hardware config override ─────────────────────────────
HW_CFG_SRC="$SCRIPT_DIR/hardware_config_8xa6000.yaml"
HW_CFG_DST="$SCRIPT_DIR/../../mediscan_v70/config/hardware_config.yaml"

if [[ -f "$HW_CFG_SRC" ]]; then
    cp "$HW_CFG_SRC" "$HW_CFG_DST"
    info "✓ hardware_config.yaml replaced with 8×A6000 version"
fi

# ── Summary ───────────────────────────────────────────────────
echo ""
echo -e "${GRN}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${GRN}║  MediScan v7.0 Setup Complete on JarvisLabs!     ║${NC}"
echo -e "${GRN}╠══════════════════════════════════════════════════╣${NC}"
echo -e "${GRN}║  Next steps:                                      ║${NC}"
echo -e "${GRN}║  1. source $VENV_DIR/bin/activate               ║${NC}"
echo -e "${GRN}║  2. bash deploy/jarvislabs/launch.sh             ║${NC}"
echo -e "${GRN}║  3. Open the JarvisLabs API endpoint on port 6006║${NC}"
echo -e "${GRN}╚══════════════════════════════════════════════════╝${NC}"
echo ""
nvidia-smi --query-gpu=index,name,memory.total,memory.free \
           --format=csv,noheader,nounits \
    | awk -F',' '{printf "  GPU%d %-20s  total=%s MB  free=%s MB\n",$1,$2,$3,$4}' \
    | tee -a "$LOG_FILE"
