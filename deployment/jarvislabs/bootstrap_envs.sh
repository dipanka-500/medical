#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

resolve_python_bin() {
  local requested="${1:-}"
  local candidate=""

  if [[ -n "$requested" ]]; then
    if command -v "$requested" >/dev/null 2>&1; then
      command -v "$requested"
      return
    fi
    echo "Requested Python interpreter not found: $requested" >&2
    exit 1
  fi

  for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      command -v "$candidate"
      return
    fi
  done

  echo "No supported Python interpreter found. Install python3.10+ first." >&2
  exit 1
}

PYTHON_BIN="$(resolve_python_bin "${PYTHON_BIN:-}")"

require_uv() {
  if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required. Install it first: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
  fi
}

ensure_venv() {
  local dir="$1"
  if [[ ! -x "$dir/.venv/bin/python" ]]; then
    (cd "$dir" && uv venv .venv --python "$PYTHON_BIN" --seed)
  fi
}

install_platform() {
  local dir="$ROOT/platform"
  ensure_venv "$dir"
  (
    cd "$dir"
    source .venv/bin/activate
    uv pip install -e .
  )
}

install_medical_llm() {
  local dir="$ROOT/medical_llm"
  ensure_venv "$dir"
  (
    cd "$dir"
    source .venv/bin/activate
    uv pip install -e ".[vllm]"
  )
}

install_general_llm_vllm() {
  local dir="$ROOT/general_llm"
  ensure_venv "$dir"
  (
    cd "$dir"
    source .venv/bin/activate
    uv pip install "vllm>=0.6.0" "openai>=1.0.0"
  )
}

install_mediscan() {
  local dir="$ROOT/mediscan_v70_sota_production/mediscan_v70"
  ensure_venv "$dir"
  (
    cd "$dir"
    source .venv/bin/activate
    uv pip install -r requirements.txt
  )
}

install_ocr() {
  local dir="$ROOT/documnet ocr"
  ensure_venv "$dir"
  (
    cd "$dir"
    source .venv/bin/activate
    uv pip install -e ".[gpu,sarvam,docsplit]" \
      "transformers>=4.50.0" \
      "accelerate>=1.2.0" \
      "safetensors>=0.4.0"
  )
}

require_uv
install_platform
install_medical_llm
install_general_llm_vllm
install_mediscan
install_ocr

echo "JarvisLabs Python environments are ready."
