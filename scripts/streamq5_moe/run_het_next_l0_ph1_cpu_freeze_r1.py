#!/usr/bin/env python3
"""Lifecycle/provenance wrapper for the immutable PH1 CPU-freezer."""
from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import os
import platform
import sys
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

os.environ.update(
    CUDA_VISIBLE_DEVICES="-1",
    HF_HUB_OFFLINE="1",
    TRANSFORMERS_OFFLINE="1",
    HF_HUB_DISABLE_TELEMETRY="1",
    USE_HUB_KERNELS="0",
    OMP_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
)

import mpmath
import numpy as np
import psutil
import safetensors
import torch


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports/streamq5_moe"
FINAL = REPORTS / "het_next_l0_ph1_cpu_freeze_r1"
FAILED = REPORTS / "het_next_l0_ph1_cpu_freeze_r1_failed_attempts"
BASE_PATH = ROOT / "scripts/streamq5_moe/generate_het_next_l0_ph1_cpu_freeze.py"
PREREG = REPORTS / "HET_NEXT_L0_PH1_SINGLE_REAL_EXPERT_PREREGISTRATION_2026-08-13.md"
DESIGN = REPORTS / "HET_NEXT_L0_PH1_IMPLEMENTATION_DESIGN_2026-08-13.md"
REPAIR = REPORTS / "HET_NEXT_L0_PH1_CPU_FREEZE_R1_REPAIR_PREREGISTRATION_2026-08-13.md"
ACK = "PH1_CPU_FREEZE_R1_AFTER_INDEPENDENT_SOURCE_GO"

EXPECTED = {
    BASE_PATH: "746a879192041dee32acb1bcb9360ce9dde6775631c0a0671312660fb71437c8",
    PREREG: "c464be6643f0301ea9f99b0e69141959a53667fa7cf9915bd540cea0a15b2b39",
    DESIGN: "4fa8a9f17b5d6c16d92c6ff1816ceda7e213e852e7fac5bfe5761c3c0338bbaf",
    ROOT / ".venv-next-ref/Lib/site-packages/transformers/models/qwen3_next/modeling_qwen3_next.py": "de40823607becdd616436e3b332f14e0c92df5495ac72ef8af027c4488b9afca",
    ROOT / ".venv-next-ref/Lib/site-packages/transformers/activations.py": "5b20c0a3625edc0001a98f09ce3c6b5baa1100e1d7ad8dee649e4d45c8468665",
    REPORTS / "port80b_t0r4_dependency_execution_lock.json": "1d08457aded09f139d25af84ba778d8e275ab5ff71967a3dc8b9a7452e6d2fae",
}
VERSIONS = {
    "python": "3.12.10",
    "torch": "2.12.1+cu132",
    "numpy": "2.2.6",
    "safetensors": "0.8.0",
    "psutil": "7.2.2",
    "mpmath": "1.3.0",
}
START_AVAILABLE_MIN = 16 * 2**30
RESERVE_MIN = 2 * 2**30
PEAK_MAX = 12 * 2**30


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode() + b"\n"


def durable(path: Path) -> None:
    with path.open("r+b" if os.name == "nt" else "rb") as handle:
        os.fsync(handle.fileno())


def atomic_new(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".inprogress")
    if path.exists() or tmp.exists():
        raise FileExistsError(path)
    with tmp.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    durable(path)


def load_base():
    spec = importlib.util.spec_from_file_location("ph1_cpu_freeze_base_immutable", BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("base_import_spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def committed(path: Path) -> bool:
    try:
        commit = json.loads((path / "commit.json").read_text())
        manifest = path / "manifest.json"
        if commit != {"kind": "ph1_cpu_freeze_r1_commit", "manifest_sha256": sha_file(manifest)}:
            return False
        rows = json.loads(manifest.read_text())["files"]
        return all(
            (path / row["name"]).is_file()
            and (path / row["name"]).stat().st_size == row["bytes"]
            and sha_file(path / row["name"]) == row["sha256"]
            for row in rows
        )
    except Exception:
        return False


def prepayload_gate() -> tuple[dict, dict]:
    observed = {str(path.relative_to(ROOT)): sha_file(path) for path in EXPECTED}
    if any(observed[str(path.relative_to(ROOT))] != expected for path, expected in EXPECTED.items()):
        raise RuntimeError("dependency_hash_drift")
    versions = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "safetensors": safetensors.__version__,
        "psutil": psutil.__version__,
        "mpmath": mpmath.__version__,
    }
    if versions != VERSIONS:
        raise RuntimeError("runtime_version_drift")
    process = psutil.Process()
    process.cpu_affinity(list(range(16)))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.mkldnn.enabled = True
    torch.set_flush_denormal(False)
    smallest = torch.from_numpy(np.asarray([1], dtype=np.uint32)).view(torch.float32)
    retained = smallest * torch.ones_like(smallest)
    witness = {
        "input_uint32": int(smallest.view(torch.uint32)[0]),
        "output_uint32": int(retained.view(torch.uint32)[0]),
    }
    runtime = {
        "versions": versions,
        "cpu_identity": platform.processor(),
        "torch_cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "affinity": process.cpu_affinity(),
        "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "mkldnn_enabled": torch.backends.mkldnn.enabled,
        "flush_denormal_false_witness": witness,
    }
    if not (
        runtime["cpu_identity"] == "Intel64 Family 6 Model 197 Stepping 2, GenuineIntel"
        and runtime["torch_cpu_capability"] == "AVX2"
        and runtime["affinity"] == list(range(16))
        and runtime["threads"] == runtime["interop_threads"] == 1
        and runtime["deterministic_algorithms"]
        and runtime["float32_matmul_precision"] == "highest"
        and runtime["mkldnn_enabled"]
        and witness == {"input_uint32": 1, "output_uint32": 1}
    ):
        raise RuntimeError("runtime_contract")
    memory = psutil.virtual_memory()
    if memory.available < START_AVAILABLE_MIN:
        raise RuntimeError("blocked_start_ram")
    return observed, {"runtime": runtime, "start_available": int(memory.available)}


def quarantine_stale() -> list[str]:
    moved = []
    for path in sorted(REPORTS.glob(FINAL.name + ".*.inprogress")):
        FAILED.mkdir(exist_ok=True)
        target = FAILED / (path.name + ".stale")
        if target.exists():
            raise FileExistsError(target)
        os.replace(path, target)
        moved.append(str(target.relative_to(ROOT)))
    return moved


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ack", required=True)
    args = parser.parse_args()
    if args.ack != ACK:
        raise SystemExit("acknowledgement mismatch")
    if FINAL.exists():
        if committed(FINAL):
            print(json.dumps({"status": "already_complete", "package": str(FINAL)}))
            return 0
        raise RuntimeError("corrupt_final_package")

    stale = quarantine_stale()
    bindings, gate = prepayload_gate()
    package = REPORTS / (FINAL.name + "." + uuid.uuid4().hex + ".inprogress")
    package.mkdir()
    base = load_base()
    base.LUT_PATH = package / "bf16_silu_lut.bin"
    base.MATH_LUT_PATH = package / "high_precision_silu_diagnostic.bin"
    base.RAW_PATH = package / "cpu_stage_freeze.safetensors"
    base.RESULT_PATH = package / "cpu_stage_freeze.json"

    original_save = base.save_file

    def cloned_save(tensors, path):
        return original_save({name: tensor.detach().clone().contiguous() for name, tensor in tensors.items()}, path)

    base.save_file = cloned_save
    original_interop = base.torch.set_num_interop_threads

    def locked_interop(value):
        if value != 1 or base.torch.get_num_interop_threads() != 1:
            raise RuntimeError("interop_contract")

    base.torch.set_num_interop_threads = locked_interop
    failure = None
    try:
        return_code = int(base.main())
        if return_code not in (0, 3):
            raise RuntimeError("unexpected_base_return")
        base_result = json.loads(base.RESULT_PATH.read_text())
        if (return_code == 0) != (base_result.get("positive") is True):
            raise RuntimeError("base_result_return_mismatch")
        gc.collect()
        process_memory = psutil.Process().memory_info()
        final_available = int(psutil.virtual_memory().available)
        peak = int(getattr(process_memory, "peak_wset", process_memory.rss))
        if final_available < RESERVE_MIN or peak > PEAK_MAX:
            raise RuntimeError("blocked_resource_after_compute")
        handoff = {
            "kind": "het_next_l0_ph1_cpu_freeze_r1_handoff",
            "base_status": base_result["status"],
            "base_positive": base_result["positive"],
            "return_code": return_code,
            "bindings": bindings,
            "wrapper_sha256": sha_file(Path(__file__)),
            "repair_prereg_sha256": sha_file(REPAIR),
            "resource": {
                **gate,
                "final_available": final_available,
                "peak_wset": peak,
                "reserve_min": RESERVE_MIN,
                "peak_max": PEAK_MAX,
            },
            "stale_quarantine_before_payload": stale,
            "device_or_compiler_opened": False,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        }
        atomic_new(package / "handoff.json", canonical(handoff))
        data_names = sorted(path.name for path in package.iterdir() if path.is_file())
        files = [
            {"name": name, "bytes": (package / name).stat().st_size, "sha256": sha_file(package / name)}
            for name in data_names
        ]
        manifest = {"kind": "ph1_cpu_freeze_r1_manifest", "files": files}
        atomic_new(package / "manifest.json", canonical(manifest))
        commit = {"kind": "ph1_cpu_freeze_r1_commit", "manifest_sha256": sha_file(package / "manifest.json")}
        atomic_new(package / "commit.json", canonical(commit))
        if not committed(package):
            raise RuntimeError("prepromotion_verification")
        os.rename(package, FINAL)
        if not committed(FINAL):
            raise RuntimeError("postpromotion_verification")
        print(
            json.dumps(
                {"status": base_result["status"], "positive": base_result["positive"], "package": str(FINAL)},
                indent=2,
            )
        )
        return return_code
    except Exception as exc:
        failure = {
            "kind": "het_next_l0_ph1_cpu_freeze_r1_failure",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "device_or_compiler_opened": False,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        }
        if package.exists():
            atomic_new(package / "failure.json", canonical(failure))
            FAILED.mkdir(exist_ok=True)
            target = FAILED / (package.name + ".failed")
            os.rename(package, target)
        raise
    finally:
        base.torch.set_num_interop_threads = original_interop


if __name__ == "__main__":
    raise SystemExit(main())
