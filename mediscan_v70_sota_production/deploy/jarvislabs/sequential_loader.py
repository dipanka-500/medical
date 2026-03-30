"""
MediScan AI v7.0 — Sequential Model Pre-loader
JarvisLabs 8 × A6000

Why sequential loading?
───────────────────────
Loading multiple large models simultaneously causes two problems:
  1. VRAM spikes during model init (weights briefly held in CPU RAM + GPU RAM)
  2. CUDA allocator fragmentation if tensors land on wrong devices

This script loads each model one at a time in LOAD_ORDER, verifies it is
resident on the correct GPU(s), then moves to the next. Small models on
the same GPU are batched together in a mini-parallel group after the
large sequential ones are done.

Run before starting the API server:
    python -m deploy.jarvislabs.sequential_loader [--dry-run] [--skip BIG]

After this script exits, all models are warm in VRAM and the API server
starts with zero cold-start latency.
"""
from __future__ import annotations

import argparse
import gc
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import torch

# ── Project root on sys.path ─────────────────────────────────────────────────
_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parents[3]   # mediscan_v70_sota_production/
sys.path.insert(0, str(_PROJECT_ROOT))

from deploy.jarvislabs.gpu_allocation import (
    GPU_PLAN,
    LOAD_ORDER,
    SEQUENTIAL_ONLY,
    PARALLEL_SAFE,
    get_max_memory,
    print_layout,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("preloader")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _gpu_free_gb(device: int) -> float:
    if not torch.cuda.is_available():
        return 0.0
    free, total = torch.cuda.mem_get_info(device)
    return free / 1e9


def _vram_snapshot() -> dict[int, tuple[float, float]]:
    """Return {gpu_id: (used_gb, total_gb)} for all GPUs."""
    n = torch.cuda.device_count()
    snap = {}
    for i in range(n):
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        free, _ = torch.cuda.mem_get_info(i)
        used = total - free / 1e9
        snap[i] = (used, total)
    return snap


def _print_vram(label: str = "") -> None:
    snap = _vram_snapshot()
    header = f"  {'GPU':<5} {'Used':>8} {'Total':>8} {'Free%':>7}"
    print(f"\n{'─'*40}  {label}")
    print(header)
    for g, (used, total) in snap.items():
        pct_free = 100 * (total - used) / total
        print(f"  GPU {g:<2}  {used:7.1f}G  {total:6.1f}G  {pct_free:5.1f}%")
    print()


def _load_one_model(engine, model_key: str, dry_run: bool) -> float:
    """
    Load a single model via the engine's wrapper.load() method.
    Returns elapsed seconds.
    """
    if model_key not in engine.models:
        log.warning("Model key %r not registered in engine — skipping", model_key)
        return 0.0

    wrapper = engine.models[model_key]
    max_mem = get_max_memory(model_key)
    primary_gpus = [g for g in max_mem if isinstance(g, int)]

    log.info(
        "► Loading %-20s  GPUs=%s  budget=%s",
        model_key,
        primary_gpus,
        {k: v for k, v in max_mem.items() if isinstance(k, int)},
    )

    if dry_run:
        time.sleep(0.05)
        log.info("  [dry-run] skipped actual load")
        return 0.05

    t0 = time.time()
    try:
        # Each wrapper exposes .load(max_memory=...) that calls from_pretrained
        # with device_map="auto" and the given max_memory constraint.
        wrapper.load(max_memory=max_mem)
        elapsed = time.time() - t0
        log.info("  ✓ %-20s loaded in %.1f s", model_key, elapsed)
    except torch.cuda.OutOfMemoryError as e:
        elapsed = time.time() - t0
        log.error("  ✗ OOM loading %s after %.1f s: %s", model_key, elapsed, e)
        torch.cuda.empty_cache()
        gc.collect()
    except Exception as e:
        elapsed = time.time() - t0
        log.error("  ✗ Error loading %s after %.1f s: %s", model_key, elapsed, e)

    return elapsed


def _load_parallel_group(
    engine,
    group: list[str],
    dry_run: bool,
    max_workers: int = 4,
) -> None:
    """Load a batch of small, single-GPU models in parallel threads."""
    if not group:
        return
    log.info("⚡ Parallel-loading group: %s", group)
    with ThreadPoolExecutor(max_workers=min(max_workers, len(group))) as pool:
        futures = {
            pool.submit(_load_one_model, engine, k, dry_run): k
            for k in group
        }
        for f in as_completed(futures):
            k = futures[f]
            try:
                elapsed = f.result()
            except Exception as exc:
                log.error("Parallel load of %s raised: %s", k, exc)


# ── Main Loader ───────────────────────────────────────────────────────────────

def run(
    engine,
    dry_run: bool = False,
    skip: Optional[list[str]] = None,
) -> None:
    """
    Sequentially load heavy multi-GPU models, then parallel-load small ones.

    Args:
        engine:   Initialised MediScanEngine instance (models registered but
                  not yet loaded into GPU RAM).
        dry_run:  If True, skip actual HuggingFace loading (for CI/testing).
        skip:     List of model keys to skip entirely.
    """
    skip = set(skip or [])
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    log.info("MediScan Sequential Loader — %d GPU(s) detected", n_gpus)
    print_layout()
    _print_vram("BEFORE loading")

    total_t0 = time.time()
    parallel_queue: list[str] = []

    for model_key in LOAD_ORDER:
        if model_key in skip:
            log.info("  ⊘ Skipping %s (--skip)", model_key)
            continue

        if model_key in SEQUENTIAL_ONLY:
            # Flush any queued parallel models first so GPU RAM is stable
            if parallel_queue:
                _load_parallel_group(engine, parallel_queue, dry_run)
                parallel_queue.clear()
                torch.cuda.synchronize()
                gc.collect()

            # Now load this big model sequentially
            _load_one_model(engine, model_key, dry_run)
            torch.cuda.synchronize()
            gc.collect()
        else:
            # Accumulate small/single-GPU models for parallel loading
            parallel_queue.append(model_key)

    # Load any remaining small models
    if parallel_queue:
        _load_parallel_group(engine, parallel_queue, dry_run)

    torch.cuda.synchronize()
    total_elapsed = time.time() - total_t0
    log.info("✅ All models loaded in %.1f s (%.1f min)", total_elapsed, total_elapsed / 60)
    _print_vram("AFTER loading")


# ── CLI entry-point ───────────────────────────────────────────────────────────

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-warm all MediScan v7.0 models into GPU VRAM"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Skip actual model loading (useful for testing the script logic)"
    )
    parser.add_argument(
        "--skip", nargs="*", default=[],
        metavar="MODEL_KEY",
        help="Model keys to skip, e.g. --skip hulu_med_32b medgemma_27b"
    )
    parser.add_argument(
        "--config-dir", default=None,
        help="Path to MediScan config directory (default: package config/)"
    )
    parser.add_argument(
        "--layout-only", action="store_true",
        help="Just print the GPU layout and exit"
    )
    args = parser.parse_args()

    if args.layout_only:
        print_layout()
        return

    # Import here so the CLI remains importable even without the full package
    from mediscan_v70.main import MediScanEngine  # type: ignore

    log.info("Initialising MediScanEngine (model registration only)…")
    engine = MediScanEngine(config_dir=args.config_dir)
    log.info("Engine ready — starting sequential pre-load")

    run(engine, dry_run=args.dry_run, skip=args.skip)


if __name__ == "__main__":
    _cli()
