"""Non-invasive environment diagnostic for the reproducible experiment setup."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LIBRARIES = ("torch", "torchvision", "ultralytics", "pandas", "matplotlib", "seaborn", "yaml")


def library_versions() -> dict[str, str | None]:
    """Return installed versions without raising when optional packages are absent."""
    versions: dict[str, str | None] = {}
    for name in LIBRARIES:
        try:
            module = importlib.import_module(name)
            versions[name] = getattr(module, "__version__", "installed (version unavailable)")
        except ImportError:
            versions[name] = None
    return versions


def torch_details() -> dict[str, Any]:
    """Collect PyTorch/CUDA data without allocating tensors or starting a workload."""
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return {"available": False, "cuda_available": False, "devices": []}

    cuda_available = bool(torch.cuda.is_available())
    devices: list[dict[str, Any]] = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "memory_gib": round(properties.total_memory / 1024**3, 2),
                    "compute_capability": f"{properties.major}.{properties.minor}",
                }
            )
    return {
        "available": True,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": cuda_available,
        "device_count": torch.cuda.device_count() if cuda_available else 0,
        "devices": devices,
    }


def nvidia_smi_details() -> dict[str, Any]:
    """Read NVIDIA hardware details when the driver utility is available."""
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"available": False, "devices": []}
    command = [
        executable,
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return {"available": False, "devices": []}
    devices = []
    for index, line in enumerate(result.stdout.splitlines()):
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 3:
            devices.append(
                {"index": index, "name": parts[0], "memory_mib": parts[1], "driver_version": parts[2]}
            )
    try:
        overview = subprocess.run([executable], check=True, capture_output=True, text=True, timeout=10)
        cuda_match = re.search(r"CUDA UMD Version:\s*([\d.]+)", overview.stdout)
        cuda_umd_version = cuda_match.group(1) if cuda_match else None
    except (OSError, subprocess.SubprocessError):
        cuda_umd_version = None
    return {"available": True, "cuda_umd_version": cuda_umd_version, "devices": devices}


def collect_report() -> dict[str, Any]:
    """Build a JSON-serializable, factual environment report."""
    libraries = library_versions()
    torch_info = torch_details()
    nvidia_info = nvidia_smi_details()
    missing = [name for name, version in libraries.items() if version is None]
    return {
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": {"version": sys.version, "executable": sys.executable},
        "libraries": libraries,
        "pytorch": torch_info,
        "nvidia_smi": nvidia_info,
        "required_packages_missing": missing,
        "training_readiness": {
            "gpu_acceleration_available": torch_info["cuda_available"],
            "assessment": (
                "GPU acceleration is available; verify free VRAM before training."
                if torch_info["cuda_available"]
                else (
                    "An NVIDIA GPU is detected, but CUDA-enabled PyTorch is not available yet. "
                    "Install the pinned environment and re-run this diagnostic."
                    if nvidia_info["devices"]
                    else "No CUDA-enabled PyTorch GPU is available; do not run full training locally."
                )
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the experiment environment without training.")
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    args = parser.parse_args()
    report = collect_report()
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Environment report written to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
