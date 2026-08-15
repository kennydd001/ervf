from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import psutil


def _nvidia_smi() -> list[dict[str, str]]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.free,driver_version,pci.bus_id",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return []
    fields = ["name", "memory_total_mib", "memory_free_mib", "driver", "pci_bus_id"]
    return [
        dict(zip(fields, (part.strip() for part in line.split(",")), strict=True))
        for line in result.stdout.splitlines()
        if line.strip()
    ]


def collect_hardware() -> dict[str, Any]:
    memory = psutil.virtual_memory()
    disk = shutil.disk_usage(Path.cwd().anchor or os.getcwd())
    payload: dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": sys.version,
        "python_executable": sys.executable,
        "logical_cpus": psutil.cpu_count(logical=True),
        "physical_cpus": psutil.cpu_count(logical=False),
        "ram_total_gib": round(memory.total / 2**30, 3),
        "ram_available_gib": round(memory.available / 2**30, 3),
        "workspace_drive_total_gib": round(disk.total / 2**30, 3),
        "workspace_drive_free_gib": round(disk.free / 2**30, 3),
        "gpus": _nvidia_smi(),
        "packages": {},
    }
    for package in ("accelerate", "huggingface-hub", "numpy", "safetensors", "transformers"):
        try:
            payload["packages"][package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            payload["packages"][package] = None
    try:
        import torch

        payload["torch"] = torch.__version__
        payload["torch_cuda_build"] = torch.version.cuda
        payload["torch_cuda_available"] = torch.cuda.is_available()
    except ImportError:
        payload["torch"] = None
        payload["torch_cuda_build"] = None
        payload["torch_cuda_available"] = False
    return payload
