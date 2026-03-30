#!/usr/bin/env python3
"""
MedAI Model Downloader — Pre-downloads ALL models locally.

After running this script, set HF_HUB_OFFLINE=1 in Docker to prevent
any runtime calls to HuggingFace.

Usage:
    # Download core models only (~25GB)
    python scripts/download_models.py --tier core

    # Download core + medical specialists (~65GB)
    python scripts/download_models.py --tier medical

    # Download everything (~300GB+, needs GPU for 70B models)
    python scripts/download_models.py --tier full

    # Download specific model(s)
    python scripts/download_models.py --models biomistral_7b,medgemma_4b

    # Custom download directory
    python scripts/download_models.py --tier core --models-dir ./my_models
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("model-downloader")

# ── Model Registry ────────────────────────────────────────────────────────
# Each entry: (key, hf_repo_id, approximate_size_gb, tier, description)

MODELS: list[tuple[str, str, float, str, str]] = [
    # ── Tier 1: Core (essential for basic functionality) ──────────────
    ("qwen2_5_7b_instruct", "Qwen/Qwen2.5-7B-Instruct", 14.0, "core",
     "General-purpose conversational LLM backbone"),
    ("biomistral_7b", "BioMistral/BioMistral-7B", 14.0, "core",
     "Fast medical routing & simple Q&A"),
    ("bge_large_en", "BAAI/bge-large-en-v1.5", 1.3, "core",
     "Text embeddings for RAG vector search"),
    ("bge_reranker_large", "BAAI/bge-reranker-large", 1.3, "core",
     "Semantic reranker used by the medical retrieval pipeline"),
    ("biomedclip", "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", 1.0, "core",
     "Zero-shot medical image classification"),
    ("firered_ocr", "FireRedTeam/FireRed-OCR", 8.0, "core",
     "Document & medical form OCR"),
    ("pathgen", "jamessyx/pathgenclip-vit-large-patch14-hf", 3.5, "core",
     "PathGen-CLIP-L HF checkpoint for pathology image analysis"),

    # ── Tier 2: Medical Specialists (recommended) ────────────────────
    ("medix_r1_2b", "MBZUAI/MediX-R1-2B", 4.0, "medical",
     "Lightweight medical vision reasoning"),
    ("medgemma_4b", "google/medgemma-4b-it", 8.0, "medical",
     "Google's medical vision-language model"),
    ("chexagent_3b", "StanfordAIMI/CheXagent-2-3b", 6.0, "medical",
     "Chest X-ray specialist with grounding"),
    ("retfound", "TJU-DRL-LAB/RETFound", 1.0, "medical",
     "Retinal/OCT image specialist"),
    ("mellama_13b", "clinicalnlplab/Me-LLaMA-13b", 26.0, "medical",
     "Conversational medical Q&A"),
    ("chatdoctor", "zl111/ChatDoctor", 14.0, "medical",
     "Patient-facing medical assistant"),
    ("hulu_med_7b", "ZJU-AI4H/Hulu-Med-7B", 14.0, "medical",
     "Multi-modal medical VLM (text/image/3D)"),
    ("pmc_llama_13b", "axiong/PMC_LLaMA_13B", 26.0, "medical",
     "Medical literature comprehension"),
    ("granite_vision_3b", "ibm-granite/granite-4.0-3b-vision", 7.0, "medical",
     "Granite Vision document extraction sidecar for tables and KVPs"),

    # ── Tier 3: Full (advanced, needs powerful hardware) ─────────────
    ("medix_r1_8b", "MBZUAI/MediX-R1-8B", 16.0, "full",
     "Medium medical vision reasoning"),
    ("medix_r1_30b", "MBZUAI/MediX-R1-30B", 60.0, "full",
     "Large medical vision reasoning (MoE)"),
    ("chexagent_8b", "StanfordAIMI/CheXagent-2-8b", 16.0, "full",
     "Chest X-ray specialist (large)"),
    ("medgemma_27b", "google/medgemma-27b-it", 54.0, "full",
     "Google's large medical vision model"),
    ("hulu_med_14b", "ZJU-AI4H/Hulu-Med-14B", 28.0, "full",
     "Multi-modal medical VLM (medium)"),
    ("hulu_med_32b", "ZJU-AI4H/Hulu-Med-32B", 64.0, "full",
     "Multi-modal medical VLM (large)"),
    ("med3dvlm", "MagicXin/Med3DVLM-Qwen-2.5-7B", 14.0, "full",
     "3D medical image analysis (CT/MRI NIfTI)"),
    ("merlin", "StanfordMIMI/Merlin", 14.0, "full",
     "3D CT volume interpretation"),
    ("radfm", "chaoyi-wu/RadFM", 14.0, "full",
     "General radiology foundation model"),
    ("deepseek_r1", "deepseek-ai/DeepSeek-R1-Distill-Llama-70B", 140.0, "full",
     "Primary reasoning engine (70B)"),
    ("meditron_70b", "epfl-llm/meditron-70b", 140.0, "full",
     "Clinical reasoning specialist (70B)"),
    ("openbiollm_70b", "aaditya/Llama3-OpenBioLLM-70B", 140.0, "full",
     "SOTA open-source medical LLM (70B)"),
    ("clinical_camel_70b", "wanglab/ClinicalCamel-70B", 140.0, "full",
     "Clinical safety validator (70B)"),
    ("med42_70b", "m42-health/Llama3-Med42-70B", 140.0, "full",
     "Diagnostic validator (70B)"),
    ("context1_20b_watch", "chromadb/context-1", 40.0, "full",
     "Context-1 watch-mode download; official harness is not public yet"),
]

# Tier hierarchy (each tier includes all tiers before it)
TIER_LEVELS = {"core": 1, "medical": 2, "full": 3}


def get_models_for_tier(tier: str) -> list[tuple[str, str, float, str, str]]:
    """Return models for the given tier (inclusive of lower tiers)."""
    level = TIER_LEVELS.get(tier, 1)
    return [m for m in MODELS if TIER_LEVELS.get(m[3], 99) <= level]


def get_models_by_keys(keys: list[str]) -> list[tuple[str, str, float, str, str]]:
    """Return specific models by key name."""
    key_set = set(keys)
    found = [m for m in MODELS if m[0] in key_set]
    missing = key_set - {m[0] for m in found}
    if missing:
        logger.warning("Unknown model keys: %s", ", ".join(sorted(missing)))
    return found


def download_model(
    repo_id: str,
    models_dir: str,
    token: str | None = None,
) -> bool:
    """Download a single model from HuggingFace Hub."""
    try:
        from huggingface_hub import snapshot_download

        logger.info("  Downloading %s ...", repo_id)
        snapshot_download(
            repo_id=repo_id,
            cache_dir=models_dir,
            token=token,
            # Resume interrupted downloads
            resume_download=True,
            # Ignore patterns that aren't needed for inference
            ignore_patterns=[
                "*.md", "*.txt", "*.gitattributes",
                "*.msgpack", "*.h5",       # TF/Flax weights (we only need PyTorch)
                "training_args.bin",
                "optimizer.pt",
                "scheduler.pt",
                "flax_model*",
                "tf_model*",
                "rust_model*",
            ],
        )
        logger.info("  Done: %s", repo_id)
        return True

    except Exception as e:
        logger.error("  FAILED: %s — %s", repo_id, e)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download MedAI models locally",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Tiers:
  core     ~25GB  — General LLM + routing + embeddings + OCR + basic vision
  medical  ~65GB  — Core + medical specialists (X-ray, retinal, pathology, Q&A)
  full    ~300GB+ — Everything including 70B models (needs powerful hardware)

Examples:
  python scripts/download_models.py --tier core
  python scripts/download_models.py --tier medical --hf-token hf_xxxxx
  python scripts/download_models.py --models biomistral_7b,medgemma_4b
""",
    )
    parser.add_argument(
        "--tier", choices=["core", "medical", "full"], default="core",
        help="Download tier (default: core)",
    )
    parser.add_argument(
        "--models", type=str, default="",
        help="Comma-separated model keys to download (overrides --tier)",
    )
    parser.add_argument(
        "--models-dir", type=str, default="./models",
        help="Local directory to store models (default: ./models)",
    )
    parser.add_argument(
        "--hf-token", type=str, default="",
        help="HuggingFace token for gated models (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all available models and exit",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be downloaded without actually downloading",
    )

    args = parser.parse_args()

    # List mode
    if args.list:
        print("\nAvailable models:\n")
        print(f"{'Key':<25} {'HuggingFace ID':<50} {'Size':>6}  {'Tier':<8} Description")
        print("-" * 130)
        for key, repo, size, tier, desc in MODELS:
            print(f"{key:<25} {repo:<50} {size:>5.1f}G  {tier:<8} {desc}")
        print()

        for tier_name in ["core", "medical", "full"]:
            tier_models = get_models_for_tier(tier_name)
            total = sum(m[2] for m in tier_models)
            print(f"  --tier {tier_name:<8}  {len(tier_models):>2} models  ~{total:.0f} GB")
        print()
        return

    # Determine which models to download
    if args.models:
        keys = [k.strip() for k in args.models.split(",") if k.strip()]
        selected = get_models_by_keys(keys)
    else:
        selected = get_models_for_tier(args.tier)

    if not selected:
        logger.error("No models selected. Use --list to see available models.")
        sys.exit(1)

    total_size = sum(m[2] for m in selected)
    logger.info(
        "Selected %d models (~%.0f GB) for download [tier=%s]",
        len(selected), total_size, args.tier,
    )
    for key, repo, size, tier, desc in selected:
        logger.info("  [%s] %s (%.1f GB) — %s", tier, repo, size, desc)

    if args.dry_run:
        logger.info("Dry run — no downloads performed.")
        return

    # Resolve models directory
    models_dir = str(Path(args.models_dir).resolve())
    os.makedirs(models_dir, exist_ok=True)
    logger.info("Download directory: %s", models_dir)

    # Set HF_HOME so all models go to the same cache
    os.environ["HF_HOME"] = models_dir

    # HuggingFace token for gated models
    token = args.hf_token or os.environ.get("HF_TOKEN", "")
    if not token:
        logger.warning(
            "No HuggingFace token provided. Some gated models (e.g., Llama, "
            "MedGemma) may fail to download. Set --hf-token or HF_TOKEN env var."
        )

    # Check huggingface_hub is installed
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        logger.error(
            "huggingface_hub not installed. Run: pip install huggingface_hub"
        )
        sys.exit(1)

    # Download
    succeeded = 0
    failed = 0
    for key, repo, size, tier, desc in selected:
        logger.info("[%d/%d] %s (%s, ~%.1f GB)",
                    succeeded + failed + 1, len(selected), key, repo, size)
        if download_model(repo, models_dir, token=token or None):
            succeeded += 1
        else:
            failed += 1

    # Summary
    logger.info("=" * 60)
    logger.info(
        "Download complete: %d succeeded, %d failed out of %d",
        succeeded, failed, len(selected),
    )
    logger.info("Models stored in: %s", models_dir)
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Set in .env:  HF_HOME=%s", models_dir)
    logger.info("  2. Rebuild:      docker compose up --build")
    logger.info("  3. Models will load from local cache (no HuggingFace calls)")
    logger.info("=" * 60)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
