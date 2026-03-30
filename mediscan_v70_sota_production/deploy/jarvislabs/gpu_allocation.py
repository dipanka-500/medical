"""
MediScan AI v7.0 — GPU Allocation Helper
JarvisLabs 8 × A6000 (48 GB each, 384 GB total)

Usage
─────
from deploy.jarvislabs.gpu_allocation import get_max_memory, GPU_PLAN

max_mem = get_max_memory("hulu_med_32b")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory=max_mem,
)

GPU layout (see hardware_config_8xa6000.yaml for details):
  GPU 0,1  → hulu_med_32b
  GPU 2,3  → medgemma_27b  (+ medix_r1_30b shares GPU 3)
  GPU 4    → hulu_med_14b + medgemma_4b + chexagent_3b
  GPU 5    → medix_r1_8b  + hulu_med_7b + med3dvlm
  GPU 6    → chexagent_8b + medix_r1_2b + radfm
  GPU 7    → pathgen + retfound + merlin + biomedclip
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

# ── Static GPU budget (identical to hardware_config_8xa6000.yaml) ─────────
# These are the MAX bytes each model may use on each GPU index.
# HuggingFace device_map="auto" respects these limits when spreading layers.
GPU_PLAN: dict[str, dict[int | str, str]] = {
    # ── Foundation VLMs ───────────────────────────────────────────
    "hulu_med_32b": {0: "43GB", 1: "43GB", "cpu": "20GB"},
    "hulu_med_14b": {4: "30GB", "cpu": "10GB"},
    "hulu_med_7b":  {5: "16GB", "cpu": "8GB"},

    # ── MedGemma ──────────────────────────────────────────────────
    "medgemma_27b": {2: "43GB", 3: "13GB", "cpu": "10GB"},
    "medgemma_4b":  {4: "10GB", "cpu": "5GB"},

    # ── MediX-R1 (Reasoning) ──────────────────────────────────────
    # medix_r1_30b is a MoE (30B total / 3B active) → ~36-40 GB resident
    "medix_r1_30b": {3: "34GB", 4: "6GB", "cpu": "10GB"},
    "medix_r1_8b":  {5: "18GB", "cpu": "5GB"},
    "medix_r1_2b":  {6: "6GB",  "cpu": "3GB"},

    # ── 3D Specialists ────────────────────────────────────────────
    "med3dvlm": {5: "16GB", "cpu": "5GB"},
    "merlin":   {7: "4GB",  "cpu": "3GB"},

    # ── Domain Specialists ────────────────────────────────────────
    "chexagent_8b": {6: "18GB", "cpu": "5GB"},
    "chexagent_3b": {4: "8GB",  "cpu": "3GB"},
    "pathgen":      {7: "5GB",  "cpu": "2GB"},
    "retfound":     {7: "2GB",  "cpu": "1GB"},
    "radfm":        {6: "9GB",  "cpu": "4GB"},

    # ── Classifiers ───────────────────────────────────────────────
    "biomedclip": {7: "2GB", "cpu": "1GB"},
}

# Loading order: largest (most VRAM) first to avoid fragmentation.
# Within each GPU group, load sequentially before the next group.
LOAD_ORDER: list[str] = [
    # GPU 0,1
    "hulu_med_32b",
    # GPU 2,3
    "medgemma_27b",
    "medix_r1_30b",     # shares GPU3 with medgemma_27b
    # GPU 4
    "hulu_med_14b",
    "medgemma_4b",
    "chexagent_3b",
    # GPU 5
    "medix_r1_8b",
    "hulu_med_7b",
    "med3dvlm",
    # GPU 6
    "chexagent_8b",
    "radfm",
    "medix_r1_2b",
    # GPU 7
    "merlin",
    "pathgen",
    "retfound",
    "biomedclip",
]

# Models that MUST be loaded sequentially (not concurrently with others)
# because they span multiple GPUs and occupy large contiguous VRAM blocks.
SEQUENTIAL_ONLY: frozenset[str] = frozenset({
    "hulu_med_32b",
    "medgemma_27b",
    "medix_r1_30b",
})

# Models safe to load in parallel (small, single-GPU)
PARALLEL_SAFE: frozenset[str] = frozenset(
    set(GPU_PLAN.keys()) - SEQUENTIAL_ONLY
)

# Which GPUs a model primarily lives on (for CUDA_VISIBLE_DEVICES hints)
PRIMARY_GPUS: dict[str, list[int]] = {
    k: [g for g in v if isinstance(g, int)]
    for k, v in GPU_PLAN.items()
}


def get_max_memory(model_key: str) -> dict[int | str, str]:
    """Return the max_memory dict for a given model key.

    Pass the returned dict directly to from_pretrained(..., max_memory=...).
    If the model key is unknown, returns an unconstrained single-GPU budget
    on GPU 7 (the least-used GPU) with CPU fallback.
    """
    if model_key in GPU_PLAN:
        return GPU_PLAN[model_key]
    # Safe fallback: GPU 7 only
    return {7: "20GB", "cpu": "10GB"}


def print_layout() -> None:
    """Print a human-readable GPU layout table."""
    # Group models by their primary GPU
    gpu_to_models: dict[int, list[tuple[str, str]]] = {}
    for model_key, mem_map in GPU_PLAN.items():
        gpu_ids = [g for g in mem_map if isinstance(g, int)]
        label = " + ".join(f"GPU{g}({mem_map[g]})" for g in gpu_ids)
        for g in gpu_ids:
            gpu_to_models.setdefault(g, []).append((model_key, mem_map[g]))

    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║  MediScan AI v7.0 — 8×A6000 GPU Layout                      ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    for gpu_id in range(8):
        models = gpu_to_models.get(gpu_id, [])
        total_gb = sum(int(m.replace("GB", "")) for _, m in models)
        model_str = ", ".join(f"{k}({m})" for k, m in models)
        print(f"║  GPU {gpu_id} ({total_gb:3d}/48 GB)  {model_str:<40}║")
    print("╚══════════════════════════════════════════════════════════════╝\n")


def load_from_yaml(yaml_path: str | Path) -> None:
    """Optionally reload GPU_PLAN from hardware_config_8xa6000.yaml."""
    global GPU_PLAN
    path = Path(yaml_path)
    if not path.exists():
        return
    with open(path) as f:
        cfg = yaml.safe_load(f)
    mem_map = cfg.get("gpu_memory_map", {})
    for model_key, raw in mem_map.items():
        parsed: dict[int | str, str] = {}
        for k, v in raw.items():
            parsed[int(k) if str(k).isdigit() else k] = str(v)
        GPU_PLAN[model_key] = parsed


if __name__ == "__main__":
    print_layout()
