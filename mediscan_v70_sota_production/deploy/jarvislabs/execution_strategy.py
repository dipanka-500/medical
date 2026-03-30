"""
MediScan AI v7.0 — Complete Parallel vs Sequential Execution Strategy
JarvisLabs 8 × A6000 (384 GB VRAM) + 2 TB Disk Storage

═══════════════════════════════════════════════════════════════════════════
SECTION 1 — JARVISLABS FORM (HOW TO FILL IT)
═══════════════════════════════════════════════════════════════════════════

Instance Name:          mediscan-v70

Storage Configuration:  2000 GB  ← you already selected this ✓
                        All HuggingFace model weights (~330 GB) stored at:
                        /home/hf_cache   (lives on this 2TB disk)
                        export HF_HOME=/home/hf_cache

HTTP Ports:             Leave BLANK — port 6006 is pre-reserved by JarvisLabs
                        and is ALREADY your public API endpoint.
                        Your FastAPI server runs on 0.0.0.0:6006.
                        (The field is for EXTRA ports beyond 6006/8889/7007/22)

Select Script:          Leave BLANK (we run setup.sh + launch.sh manually)

Select File Storage:    Select the 2TB volume you created
                        (this mounts as /home on the instance)

═══════════════════════════════════════════════════════════════════════════
SECTION 2 — VRAM BUDGET: ALL 16 MODELS FIT IN 384 GB
═══════════════════════════════════════════════════════════════════════════

With 8 × A6000 (48 GB each = 384 GB total), ALL 16 models stay permanently
resident in VRAM. No mid-inference load/unload is ever needed.

Model weight sizes (bfloat16 = 2 bytes/param):
  hulu_med_32b      32B  → ~64 GB   GPU 0,1
  medgemma_27b      27B  → ~54 GB   GPU 2,3
  medix_r1_30b      30B  → ~60 GB   GPU 3,4  (MoE: all params resident)
  hulu_med_14b      14B  → ~28 GB   GPU 4
  medix_r1_8b        8B  → ~16 GB   GPU 5
  hulu_med_7b        7B  → ~14 GB   GPU 5
  chexagent_8b       8B  → ~16 GB   GPU 6
  med3dvlm           7B  → ~14 GB   GPU 5
  medgemma_4b        4B  →  ~8 GB   GPU 4
  medix_r1_2b        2B  →  ~4 GB   GPU 6
  chexagent_3b       3B  →  ~6 GB   GPU 4
  radfm            ~3.5B →  ~7 GB   GPU 6
  pathgen          1.6B  →  ~3 GB   GPU 7
  merlin           ~1B   →  ~2 GB   GPU 7
  retfound         ViT   →  ~0.3GB  GPU 7
  biomedclip       ~0.4B →  ~1 GB   GPU 7
  ─────────────────────────────────────────
  TOTAL weights         ~297 GB
  + KV cache (inference) ~40 GB
  + CUDA/OS overhead     ~20 GB
  ─────────────────────────────────────────
  TOTAL                 ~357 GB  ✓ fits in 384 GB

═══════════════════════════════════════════════════════════════════════════
SECTION 3 — LOADING STRATEGY (SEQUENTIAL AT STARTUP, ONCE)
═══════════════════════════════════════════════════════════════════════════

Loading is done ONCE at startup via sequential_loader.py.
After that, all models remain in VRAM forever (no eviction, no reloading).

Phase A — SEQUENTIAL (one at a time, GPU memory synced between each):
  Reason: these span 2 GPUs each. Loading two simultaneously causes
  VRAM spikes that exceed available memory during the init phase.

  Step 1: hulu_med_32b   (64 GB across GPU 0+1)   ~8-15 min first run
  Step 2: medgemma_27b   (54 GB across GPU 2+3)   ~6-12 min first run
  Step 3: medix_r1_30b   (60 GB across GPU 3+4)   ~7-13 min first run

Phase B — PARALLEL (4 worker threads, grouped by target GPU):
  Reason: single-GPU models with no overlap, safe to load concurrently.

  Group B1 (GPU 4):  hulu_med_14b + medgemma_4b + chexagent_3b
  Group B2 (GPU 5):  medix_r1_8b  + hulu_med_7b  + med3dvlm
  Group B3 (GPU 6):  chexagent_8b + radfm         + medix_r1_2b
  Group B4 (GPU 7):  merlin       + pathgen        + retfound + biomedclip

  → All 4 groups load IN PARALLEL → saves ~15 min vs sequential

Total startup time (cached weights on 2TB disk):
  Phase A sequential:  ~5-8 min   (reading from disk into VRAM)
  Phase B parallel:    ~3-5 min   (all groups loading simultaneously)
  ─────────────────────────────────────────────────────────────
  TOTAL (after first download): ~8-13 min to be fully ready

═══════════════════════════════════════════════════════════════════════════
SECTION 4 — INFERENCE STRATEGY (PER REQUEST)
═══════════════════════════════════════════════════════════════════════════

Once all models are in VRAM, INFERENCE is always parallel.
The ParallelExecutor runs ThreadPoolExecutor(max_workers=8).
Models on DIFFERENT GPUs run truly in parallel (zero contention).
Models on the SAME GPU are serialised by the CUDA scheduler automatically.

─── A. Document OCR Pipeline (lightweight, CPU-dominant) ────────────────

  Step 1 [CPU, sequential] — HandwritingClassifier (ResNet18, 45 MB)
    → classify: HANDWRITTEN / PRINTED / MIXED

  Step 2 [parallel] — Routing + Language Detection
    → RoutePlanner decides primary/secondary OCR backend

  Step 3 [parallel threads] — OCR backends (CPU or API call):
    Handwritten English  → chandra_command (local)  ‖  surya_command
    Handwritten Indic    → sarvam_vision (API)      ‖  surya_command
    Printed English      → firered_backend (local)  ‖  surya_command
    Printed Indic        → surya_command            ‖  sarvam_vision
    Structure-heavy      → + granite_vision (3B VLM) added as secondary

  Step 4 [parallel] — Enrichers (if needed):
    HIGH layout complexity → layoutlmv3 placeholder
    Table-heavy            → table_transformer placeholder

  GPU usage: NONE (OCR runs on CPU/API). Granite Vision uses ~6 GB GPU if enabled.
  No conflict with MediScan models.

─── B. MediScan v7.0 Pipeline (per imaging modality) ───────────────────

All models listed below are ALREADY IN VRAM. Calls are pure inference,
no loading/unloading, no waiting. parallel_executor runs them all at once.

MODALITY         PARALLEL MODELS (all fire simultaneously)           GPUs
────────────────────────────────────────────────────────────────────────
xray / CXR       chexagent_8b  ‖  medgemma_4b  ‖  hulu_med_7b      6 ‖ 4 ‖ 5
                 biomedclip    ‖  medix_r1_8b   ‖  radfm             7 ‖ 5 ‖ 6
                 (chexagent_3b as verifier — GPU 4)

ct (2D slices)   hulu_med_14b  ‖  medgemma_4b  ‖  hulu_med_7b      4 ‖ 4 ‖ 5
                 biomedclip    ‖  medix_r1_8b   ‖  radfm             7 ‖ 5 ‖ 6

ct (3D NIfTI)    hulu_med_14b  ‖  med3dvlm     ‖  merlin             4 ‖ 5 ‖ 7
                 biomedclip    ‖  medix_r1_8b   ‖  radfm             7 ‖ 5 ‖ 6
                 ↳ Three 3D models run TRULY in parallel (GPU 4, 5, 7)

mri              hulu_med_14b  ‖  medgemma_4b  ‖  hulu_med_7b      4 ‖ 4 ‖ 5
                 med3dvlm      ‖  biomedclip    ‖  medix_r1_8b       5 ‖ 7 ‖ 5
                 radfm (GPU 6)

pathology        pathgen       ‖  medgemma_4b  ‖  hulu_med_7b      7 ‖ 4 ‖ 5
                 biomedclip    ‖  medix_r1_8b                        7 ‖ 5
                 ↳ All on different GPUs → true parallelism

fundoscopy       medgemma_4b   ‖  retfound     ‖  hulu_med_7b      4 ‖ 7 ‖ 5
                 biomedclip    ‖  medix_r1_8b                        7 ‖ 5

video            hulu_med_7b   ‖  medix_r1_2b                       5 ‖ 6

ultrasound       hulu_med_7b   ‖  medgemma_4b  ‖  medix_r1_2b      5 ‖ 4 ‖ 6
                 biomedclip                                            7

CRITICAL query   hulu_med_32b  ‖  medgemma_27b ‖  medix_r1_30b    0-1‖2-3‖3-4
(safety route)   ↳ Big models activated only for critical/complex cases

─── C. Medical LLM (Text-only reasoning, no image) ─────────────────────

  Simple query     → medix_r1_2b   (GPU 6, ~2s)   ← fastest
  Standard query   → medix_r1_8b   (GPU 5, ~5s)   ← default
  Complex/critical → medix_r1_30b  (GPU 3-4, ~15s) ← most thorough
  Report drafting  → hulu_med_14b  (GPU 4, ~8s)
  Rare escalation  → hulu_med_32b  (GPU 0-1, ~25s) ← only if forced

  MedicalReasoner automatically picks size via confidence threshold.

═══════════════════════════════════════════════════════════════════════════
SECTION 5 — WHEN TO OFFLOAD / EVICT (ALMOST NEVER ON 8×A6000)
═══════════════════════════════════════════════════════════════════════════

With 384 GB VRAM and ~357 GB used → ~27 GB free headroom.

NEVER offload unless:
  • torch.cuda.OutOfMemoryError is thrown mid-inference
  • circuit_breaker.py triggers OOM path → clears cache, continues
  • You run hulu_med_32b AND medgemma_27b AND medix_r1_30b simultaneously
    on the same request (safety escalation path) — possible OOM spike

If OOM does occur:
  1. ParallelExecutor catches OutOfMemoryError, calls torch.cuda.empty_cache()
  2. Does NOT retry (OOM = structural, not transient)
  3. Returns partial results from the models that succeeded
  4. FusionLayer weighs surviving model outputs

To free VRAM manually if needed (e.g., to load a new experimental model):
  engine.models['hulu_med_32b'].unload()   # wrapper must implement .unload()
  torch.cuda.empty_cache()
  gc.collect()

═══════════════════════════════════════════════════════════════════════════
SECTION 6 — CONCURRENT REQUEST HANDLING
═══════════════════════════════════════════════════════════════════════════

uvicorn runs with 1 worker (single process, all models in-process).
Multiple HTTP requests are handled by asyncio concurrency.

Request 1 (CXR)       → fires: GPU 4,5,6,7  (chexagent, medix_r1, biomedclip…)
Request 2 (Pathology) → fires: GPU 4,5,7    (pathgen, medgemma, medix_r1…)
                                             ↑ GPU 4,5,7 are shared across requests
                                             CUDA queues them automatically.
                                             No deadlock risk (no locks on GPU level).

Concurrency limit: set MEDISCAN_MAX_CONCURRENT_REQUESTS=4 to throttle
to 4 simultaneous requests if VRAM headroom shrinks during heavy load.

═══════════════════════════════════════════════════════════════════════════
SECTION 7 — 2TB DISK STORAGE USAGE MAP
═══════════════════════════════════════════════════════════════════════════

Mounted at:  /home  (JarvisLabs persistent volume, survives pause/resume)

/home/hf_cache/          ← HuggingFace model weights (~330 GB)
  hub/models--ZJU-AI4H/  ← HuluMed family
  hub/models--google/    ← MedGemma family
  hub/models--MBZUAI/    ← MediX-R1 family
  hub/models--MagicXin/  ← Med3DVLM
  hub/models--StanfordMIMI/  ← Merlin
  hub/models--StanfordAIMI/  ← CheXagent
  hub/models--PathGen/   ← PathGen
  hub/models--TJU-DRL-LAB/  ← RETFound
  hub/models--chaoyi-wu/ ← RadFM
  hub/models--microsoft/ ← BiomedCLIP

/home/venv/              ← Python virtual environment (~10 GB)
/home/mediscan_v70_sota_production/  ← project code (symlinked or git cloned)
/home/data/rag/          ← RAG knowledge base (PubMed, WHO, radiology guidelines)
/home/logs/              ← Server + preload logs
/home/offload/           ← Emergency disk offload (disabled by default)

Remaining: 2000 - 330 - 10 - ~5 (RAG/logs) = ~1655 GB free for patient data,
           DICOM files, and future model additions.
"""

# ── Summary constants (importable by other scripts) ──────────────────────────

# Models that CANNOT share a loading thread (multi-GPU, VRAM spike risk)
SEQUENTIAL_LOAD = ["hulu_med_32b", "medgemma_27b", "medix_r1_30b"]

# Parallel load groups (by primary GPU — safe to load concurrently)
PARALLEL_LOAD_GROUPS = [
    ["hulu_med_14b", "medgemma_4b", "chexagent_3b"],    # GPU 4
    ["medix_r1_8b",  "hulu_med_7b", "med3dvlm"],         # GPU 5
    ["chexagent_8b", "radfm",       "medix_r1_2b"],      # GPU 6
    ["merlin", "pathgen", "retfound", "biomedclip"],     # GPU 7
]

# Per-modality parallel inference sets (all fire simultaneously)
PARALLEL_INFERENCE = {
    "xray":       ["chexagent_8b", "medgemma_4b", "hulu_med_7b",
                   "biomedclip", "medix_r1_8b", "radfm", "chexagent_3b"],
    "ct_2d":      ["hulu_med_14b", "medgemma_4b", "hulu_med_7b",
                   "biomedclip", "medix_r1_8b", "radfm"],
    "ct_3d":      ["hulu_med_14b", "med3dvlm", "merlin",
                   "biomedclip", "medix_r1_8b", "radfm"],
    "mri":        ["hulu_med_14b", "medgemma_4b", "hulu_med_7b",
                   "med3dvlm", "biomedclip", "medix_r1_8b", "radfm"],
    "pathology":  ["pathgen", "medgemma_4b", "hulu_med_7b",
                   "biomedclip", "medix_r1_8b"],
    "fundoscopy": ["medgemma_4b", "retfound", "hulu_med_7b",
                   "biomedclip", "medix_r1_8b"],
    "ultrasound": ["hulu_med_7b", "medgemma_4b", "medix_r1_2b", "biomedclip"],
    "video":      ["hulu_med_7b", "medix_r1_2b"],
    "critical":   ["hulu_med_32b", "medgemma_27b", "medix_r1_30b"],  # safety escalation
}

# OCR routing (CPU-only — no conflict with GPU models)
OCR_ROUTING = {
    ("handwritten", "english"): {
        "primary": "chandra_command",
        "secondary": ["surya_command", "sarvam_vision"],
        "gpu_used": False,
    },
    ("handwritten", "indic"): {
        "primary": "sarvam_vision",
        "secondary": ["chandra_command", "surya_command"],
        "gpu_used": False,
    },
    ("printed", "english"): {
        "primary": "firered_backend",
        "secondary": ["surya_command", "doctr"],
        "gpu_used": False,
    },
    ("printed", "indic"): {
        "primary": "surya_command",
        "secondary": ["sarvam_vision", "doctr"],
        "gpu_used": False,
    },
    ("structure_heavy", "any"): {
        "primary": "firered_backend",
        "secondary": ["granite_vision"],     # Granite Vision 3B on GPU 7 (~6 GB)
        "gpu_used": True,                    # GPU 7 briefly
        "gpu_budget": "6GB",
    },
}

# Medical LLM text-only routing (no image)
MEDICAL_LLM_ROUTING = {
    "simple":    ("medix_r1_2b",  5, "GPU 6"),    # <5s
    "standard":  ("medix_r1_8b",  15, "GPU 5"),   # ~5-15s
    "complex":   ("medix_r1_30b", 60, "GPU 3,4"), # ~15-60s
    "report":    ("hulu_med_14b", 30, "GPU 4"),   # ~10-30s
    "critical":  ("hulu_med_32b", 90, "GPU 0,1"), # ~30-90s, rare
}


if __name__ == "__main__":
    print(__doc__)
