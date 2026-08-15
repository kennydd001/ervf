"""Shared, dependency-light helpers for the additive PRO research pack.

This module deliberately does not import CuPy or the model runtime at import time.
That lets preflight checks verify that the GPU is free before a CUDA context is
created and lets the independent verifier inspect result files on CPU-only hosts.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

REPO = Path(__file__).resolve().parents[1]
PRO = REPO / "pro_research"
RESULTS = PRO / "results"
HISTORY = RESULTS / "history"
LOGS = RESULTS / "logs"
DEFAULT_MODEL_NAME = "nemotron_3_5_lightning_v35"


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def timestamp_slug() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def ensure_dirs() -> None:
    for path in (RESULTS, HISTORY, LOGS):
        path.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def archive_existing(path: Path) -> Path | None:
    """Move an existing result into the append-only history directory."""
    ensure_dirs()
    if not path.exists():
        return None
    suffix = path.suffix or ".dat"
    archived = HISTORY / f"{path.stem}__{timestamp_slug()}{suffix}"
    counter = 1
    while archived.exists():
        archived = HISTORY / f"{path.stem}__{timestamp_slug()}_{counter}{suffix}"
        counter += 1
    shutil.move(str(path), str(archived))
    return archived


def write_json_atomic(path: Path, payload: Any, *, archive: bool = True) -> None:
    ensure_dirs()
    path.parent.mkdir(parents=True, exist_ok=True)
    if archive:
        archive_existing(path)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as fh:
        json.dump(payload, fh, indent=2, sort_keys=False, allow_nan=False)
        fh.write("\n")
        tmp = Path(fh.name)
    os.replace(tmp, path)


def write_text_atomic(path: Path, text: str, *, archive: bool = True) -> None:
    ensure_dirs()
    path.parent.mkdir(parents=True, exist_ok=True)
    if archive:
        archive_existing(path)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as fh:
        fh.write(text)
        if text and not text.endswith("\n"):
            fh.write("\n")
        tmp = Path(fh.name)
    os.replace(tmp, path)


def run_text(command: list[str], *, timeout: int = 30) -> str:
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"ERROR: {exc}"
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    if proc.returncode and err:
        return f"rc={proc.returncode}; stdout={out}; stderr={err}"
    return out or err


def git_head() -> str | None:
    out = run_text(["git", "-C", str(REPO), "rev-parse", "HEAD"])
    if out.startswith("ERROR") or out.startswith("rc="):
        return None
    return out.splitlines()[0].strip() or None


def git_status() -> list[str]:
    out = run_text(["git", "-C", str(REPO), "status", "--short"])
    if out.startswith("ERROR") or out.startswith("rc=") or not out:
        return []
    return out.splitlines()


def gpu_processes() -> list[str]:
    out = run_text(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if out.startswith("ERROR") or out.startswith("rc=") or not out:
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def require_gpu_free() -> None:
    processes = gpu_processes()
    if processes:
        joined = "\n  ".join(processes)
        raise RuntimeError(
            "Another process currently owns a CUDA context. Stop cleanly and retry; "
            f"the runner will not kill it.\n  {joined}"
        )


def nvidia_snapshot() -> str:
    return run_text(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.used,temperature.gpu,"
            "power.draw,clocks.sm,clocks.mem,pstate",
            "--format=csv,noheader",
        ]
    )


def model_dir() -> Path:
    name = os.environ.get("LS_MODEL_DIR", DEFAULT_MODEL_NAME)
    path = Path(name)
    if not path.is_absolute():
        path = REPO / "models" / path
    return path


def require_model_dir() -> Path:
    path = model_dir()
    if not path.exists():
        raise FileNotFoundError(
            f"Model directory not found: {path}. Set LS_MODEL_DIR to the local "
            "Nemotron 3.5 Lightning model directory."
        )
    return path


def environment_snapshot(extra_files: Iterable[Path] = ()) -> dict[str, Any]:
    files: dict[str, str | None] = {}
    for path in extra_files:
        try:
            files[str(path.relative_to(REPO))] = sha256_file(path)
        except (OSError, ValueError):
            files[str(path)] = None
    return {
        "created_utc": utc_now(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "git_head": git_head(),
        "git_status": git_status(),
        "model_dir": str(model_dir()),
        "nvidia_smi": nvidia_snapshot(),
        "source_hashes": files,
    }


def percentiles(samples: list[float]) -> dict[str, float | int | None]:
    if not samples:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "p99": None, "max": None}
    import numpy as np

    a = np.asarray(samples, dtype=np.float64)
    return {
        "count": int(a.size),
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
    }


def first_divergence(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def geometric_mean(values: Iterable[float]) -> float | None:
    import math

    vals = [float(v) for v in values if v > 0.0 and math.isfinite(float(v))]
    if not vals:
        return None
    return float(math.exp(sum(math.log(v) for v in vals) / len(vals)))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def result_path(name: str) -> Path:
    ensure_dirs()
    return RESULTS / name
