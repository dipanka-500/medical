"""
MediScan AI v7.0 — Real-time GPU Monitor
JarvisLabs 8 × A6000

Prints a live table of per-GPU VRAM usage and temperature, refreshing
every N seconds. Run this in a second JupyterLab terminal or SSH session
while the server is starting up.

Usage:
    python deploy/jarvislabs/monitor_gpus.py              # refresh every 5s
    python deploy/jarvislabs/monitor_gpus.py --interval 2  # every 2s
    python deploy/jarvislabs/monitor_gpus.py --once         # single snapshot
"""
from __future__ import annotations

import argparse
import os
import subprocess
import time
from typing import NamedTuple

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

# Expected models per GPU (from gpu_allocation.py)
_GPU_TENANTS: dict[int, list[str]] = {
    0: ["hulu_med_32b"],
    1: ["hulu_med_32b"],
    2: ["medgemma_27b"],
    3: ["medgemma_27b", "medix_r1_30b"],
    4: ["medix_r1_30b", "hulu_med_14b", "medgemma_4b", "chexagent_3b"],
    5: ["medix_r1_8b", "hulu_med_7b", "med3dvlm"],
    6: ["chexagent_8b", "radfm", "medix_r1_2b"],
    7: ["merlin", "pathgen", "retfound", "biomedclip"],
}


class GPUStat(NamedTuple):
    index: int
    name: str
    used_mb: int
    total_mb: int
    temp_c: int
    util_pct: int


def _query_nvidia_smi() -> list[GPUStat]:
    """Use nvidia-smi to get GPU stats (works without torch)."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,temperature.gpu,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []

    stats = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        stats.append(GPUStat(
            index=int(parts[0]),
            name=parts[1],
            used_mb=int(parts[2]),
            total_mb=int(parts[3]),
            temp_c=int(parts[4]),
            util_pct=int(parts[5]),
        ))
    return stats


def _bar(used: int, total: int, width: int = 20) -> str:
    frac = used / total if total else 0
    filled = int(frac * width)
    bar = "█" * filled + "░" * (width - filled)
    return bar


def _colour(used_mb: int, total_mb: int) -> str:
    """ANSI colour based on fill %."""
    pct = 100 * used_mb / total_mb if total_mb else 0
    if pct >= 90:
        return "\033[91m"   # red
    if pct >= 70:
        return "\033[93m"   # yellow
    return "\033[92m"       # green


RESET = "\033[0m"
BOLD  = "\033[1m"
CYAN  = "\033[96m"


def _render(stats: list[GPUStat]) -> str:
    lines = []
    ts = time.strftime("%H:%M:%S")
    lines.append(f"{BOLD}{CYAN}MediScan v7.0 — GPU Monitor  [{ts}]{RESET}")
    lines.append(
        f"{'GPU':<5} {'Name':<24} {'Used/Total (MB)':>18}  {'Bar':^22}  {'Tmp':>4}  {'Util':>5}  Models"
    )
    lines.append("─" * 110)
    for s in stats:
        col = _colour(s.used_mb, s.total_mb)
        bar = _bar(s.used_mb, s.total_mb)
        tenants = ", ".join(_GPU_TENANTS.get(s.index, ["?"]))
        pct = 100 * s.used_mb / s.total_mb if s.total_mb else 0
        lines.append(
            f"GPU{s.index:<2} {s.name:<24} "
            f"{s.used_mb:>7}/{s.total_mb:<7} MB  "
            f"{col}{bar}{RESET} {pct:5.1f}%  "
            f"{s.temp_c:>3}°C  {s.util_pct:>4}%  "
            f"{tenants}"
        )

    # Totals
    if stats:
        total_used  = sum(s.used_mb  for s in stats)
        total_total = sum(s.total_mb for s in stats)
        col = _colour(total_used, total_total)
        lines.append("─" * 110)
        lines.append(
            f"{'TOTAL':<5} {'':24} "
            f"{total_used:>7}/{total_total:<7} MB  "
            f"{col}{_bar(total_used, total_total)}{RESET} "
            f"{100*total_used/total_total:.1f}%"
        )
    return "\n".join(lines)


def _clear():
    os.system("cls" if os.name == "nt" else "clear")


def run(interval: float = 5.0, once: bool = False) -> None:
    while True:
        stats = _query_nvidia_smi()
        if not once:
            _clear()
        print(_render(stats))
        if once:
            break
        print(f"\n  Refreshing every {interval}s — Ctrl+C to stop")
        time.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time GPU monitor for MediScan")
    parser.add_argument("--interval", type=float, default=5.0, help="Refresh interval in seconds")
    parser.add_argument("--once", action="store_true", help="Print a single snapshot and exit")
    args = parser.parse_args()

    try:
        run(interval=args.interval, once=args.once)
    except KeyboardInterrupt:
        print("\nMonitor stopped.")


if __name__ == "__main__":
    main()
