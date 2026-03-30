"""
Granite Vision 4.0 3B — vLLM server launcher.

Downloads the custom serving script from the model repo if not present,
then launches the vLLM OpenAI-compatible server.

Supports two modes:
  1. Full Merge (default): Fastest inference, merges LoRA at startup
  2. Native LoRA: Flexible, allows hot-swapping adapters at runtime
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_ID = "ibm-granite/granite-4.0-3b-vision"


def _download_serving_scripts() -> None:
    """Download Granite-specific vLLM serving scripts from HuggingFace."""
    hf_home = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    offline = os.getenv("HF_HUB_OFFLINE", "0") in {"1", "true"}

    scripts = ["granite4_vision.py", "start_granite4_vision_server.py"]
    all_present = all(Path(s).exists() for s in scripts)

    if all_present:
        logger.info("Granite serving scripts already present")
        return

    if offline:
        # Check in model cache
        cache_dir = Path(hf_home) / "hub" / f"models--{MODEL_ID.replace('/', '--')}" / "snapshots"
        if cache_dir.exists():
            for snapshot in cache_dir.iterdir():
                for script in scripts:
                    src = snapshot / script
                    if src.exists() and not Path(script).exists():
                        import shutil
                        shutil.copy2(src, script)
                        logger.info("Copied %s from model cache", script)
        return

    try:
        from huggingface_hub import hf_hub_download
        for script in scripts:
            if not Path(script).exists():
                hf_hub_download(repo_id=MODEL_ID, filename=script, local_dir=".")
                logger.info("Downloaded %s", script)
    except Exception as e:
        logger.warning("Could not download serving scripts: %s", e)


def main():
    parser = argparse.ArgumentParser(description="Granite Vision vLLM Server")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8005)
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--enable-lora", action="store_true", default=False)
    parser.add_argument("--max-lora-rank", type=int, default=256)
    parser.add_argument("--hf-overrides", default="")
    parser.add_argument("--default-mm-loras", default="")
    args = parser.parse_args()

    _download_serving_scripts()

    # Build vLLM command
    granite_script = Path("start_granite4_vision_server.py")
    if granite_script.exists():
        cmd = [
            sys.executable, str(granite_script),
            "--model", args.model,
            "--host", args.host,
            "--port", str(args.port),
        ]
        if args.trust_remote_code:
            cmd.append("--trust_remote_code")
        if args.hf_overrides:
            cmd.extend(["--hf-overrides", args.hf_overrides])
        if args.enable_lora:
            cmd.extend(["--enable-lora", "--max-lora-rank", str(args.max_lora_rank)])
        if args.default_mm_loras:
            cmd.extend(["--default-mm-loras", args.default_mm_loras])
    else:
        # Fallback: direct vLLM launch
        logger.warning("Granite serving script not found, using direct vLLM launch")
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", args.model,
            "--host", args.host,
            "--port", str(args.port),
            "--trust-remote-code",
            "--dtype", "bfloat16",
            "--max-model-len", "4096",
        ]

    logger.info("Starting Granite Vision server: %s", " ".join(cmd))
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
