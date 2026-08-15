#!/usr/bin/env python3
"""PH1 CPU-freeze R2: strict package lifecycle around immutable R1/base."""
from __future__ import annotations

import ctypes
import gc
import importlib.util
import json
import os
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports/streamq5_moe"
R1_PATH = ROOT / "scripts/streamq5_moe/run_het_next_l0_ph1_cpu_freeze_r1.py"
R2_PREREG = REPORTS / "HET_NEXT_L0_PH1_CPU_FREEZE_R2_LIFECYCLE_PREREGISTRATION_2026-08-13.md"
AUTH_LOCK = REPORTS / "het_next_l0_ph1_cpu_freeze_r2_authorization_lock.json"
FINAL = REPORTS / "het_next_l0_ph1_cpu_freeze_r2"
FAILED = REPORTS / "het_next_l0_ph1_cpu_freeze_r2_failed_attempts"
ACK = "PH1_CPU_FREEZE_R2_AFTER_LIFECYCLE_GO"

EXPECTED_STATIC = {
    "r1_runner": (
        R1_PATH,
        "df824adc9072bfadec3a53570b25a531cf69493e89cde59e086dc185ce888987",
    ),
    "r1_repair": (
        REPORTS / "HET_NEXT_L0_PH1_CPU_FREEZE_R1_REPAIR_PREREGISTRATION_2026-08-13.md",
        "61037a7d5b61cc02f818a82099ad20753cc05032afc9cb3e27ec701ca9a0a975",
    ),
    "scientific_base": (
        ROOT / "scripts/streamq5_moe/generate_het_next_l0_ph1_cpu_freeze.py",
        "746a879192041dee32acb1bcb9360ce9dde6775631c0a0671312660fb71437c8",
    ),
}


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("import_spec:" + name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


r1 = load(R1_PATH, "ph1_cpu_freeze_r1_immutable")


def durable_move(source: Path, destination: Path) -> str:
    if not source.exists() or destination.exists():
        raise FileExistsError((source, destination))
    if source.drive.lower() != destination.drive.lower():
        raise RuntimeError("cross_volume_move")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        move = ctypes.windll.kernel32.MoveFileExW
        move.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32]
        move.restype = ctypes.c_int
        if not move(str(source), str(destination), 0x8):
            raise ctypes.WinError(ctypes.get_last_error())
        return "MoveFileExW_MOVEFILE_WRITE_THROUGH"
    os.rename(source, destination)
    descriptor = os.open(str(destination.parent), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return "rename_plus_parent_fsync"


def exact_committed(path: Path) -> bool:
    try:
        commit_path = path / "commit.json"
        manifest_path = path / "manifest.json"
        commit = json.loads(commit_path.read_text())
        manifest = json.loads(manifest_path.read_text())
        rows = manifest["files"]
        names = [row["name"] for row in rows]
        if (
            manifest.get("kind") != "ph1_cpu_freeze_r2_manifest"
            or len(names) != len(set(names))
            or any(Path(name).name != name or name in ("manifest.json", "commit.json") for name in names)
        ):
            return False
        exact_names = set(names) | {"manifest.json", "commit.json"}
        if {entry.name for entry in path.iterdir()} != exact_names:
            return False
        if not all(
            (path / row["name"]).is_file()
            and (path / row["name"]).stat().st_size == row["bytes"]
            and r1.sha_file(path / row["name"]) == row["sha256"]
            for row in rows
        ):
            return False
        expected_commit = {
            "kind": "ph1_cpu_freeze_r2_commit",
            "manifest_sha256": r1.sha_file(manifest_path),
            "handoff_sha256": r1.sha_file(path / "handoff.json"),
            "base_result_sha256": r1.sha_file(path / "cpu_stage_freeze.json"),
        }
        return commit == expected_commit
    except Exception:
        return False


def authorization_gate() -> dict:
    lock = json.loads(AUTH_LOCK.read_text())
    required_keys = {
        "kind",
        "execution_open",
        "audit_token",
        "runner_sha256",
        "r2_prereg_sha256",
        "r1_runner_sha256",
        "r1_repair_sha256",
        "scientific_base_sha256",
    }
    if set(lock) != required_keys:
        raise RuntimeError("authorization_schema")
    if not (
        lock["kind"] == "het_next_l0_ph1_cpu_freeze_r2_authorization"
        and lock["execution_open"] is True
        and lock["audit_token"] == ACK
        and lock["runner_sha256"] == r1.sha_file(Path(__file__))
        and lock["r2_prereg_sha256"] == r1.sha_file(R2_PREREG)
    ):
        raise RuntimeError("authorization_state")
    observed = {}
    for key, (path, expected) in EXPECTED_STATIC.items():
        value = r1.sha_file(path)
        observed[key] = value
        if value != expected or lock[key + "_sha256"] != expected:
            raise RuntimeError("authorization_binding:" + key)
    return {"lock_sha256": r1.sha_file(AUTH_LOCK), "observed": observed}


def quarantine_existing() -> list[dict]:
    rows = []
    candidates = []
    if FINAL.exists() and not exact_committed(FINAL):
        candidates.append((FINAL, "corrupt_final"))
    candidates.extend((path, "stale_temp") for path in sorted(REPORTS.glob(FINAL.name + ".*.inprogress")))
    for source, reason in candidates:
        target = FAILED / (source.name + "." + uuid.uuid4().hex + ".quarantined")
        method = durable_move(source, target)
        rows.append({"source": str(source.relative_to(ROOT)), "target": str(target.relative_to(ROOT)), "reason": reason, "method": method})
    return rows


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ack", required=True)
    args = parser.parse_args()
    if args.ack != ACK:
        raise SystemExit("acknowledgement mismatch")

    authorization = authorization_gate()  # before every filesystem mutation/payload read
    if FINAL.exists() and exact_committed(FINAL):
        print(json.dumps({"status": "already_complete", "package": str(FINAL)}))
        return 0
    quarantine = quarantine_existing()
    bindings, resource_start = r1.prepayload_gate()  # still before payload

    package = REPORTS / (FINAL.name + "." + uuid.uuid4().hex + ".inprogress")
    package.mkdir()
    base = r1.load_base()
    base.LUT_PATH = package / "bf16_silu_lut.bin"
    base.MATH_LUT_PATH = package / "high_precision_silu_diagnostic.bin"
    base.RAW_PATH = package / "cpu_stage_freeze.safetensors"
    base.RESULT_PATH = package / "cpu_stage_freeze.json"
    original_save = base.save_file
    original_interop = base.torch.set_num_interop_threads

    def cloned_save(tensors, path):
        return original_save({name: tensor.detach().clone().contiguous() for name, tensor in tensors.items()}, path)

    def locked_interop(value):
        if value != 1 or base.torch.get_num_interop_threads() != 1:
            raise RuntimeError("interop_contract")

    base.save_file = cloned_save
    base.torch.set_num_interop_threads = locked_interop
    original_error = None
    try:
        return_code = int(base.main())
        if return_code not in (0, 3):
            raise RuntimeError("unexpected_base_return")
        base_result = json.loads(base.RESULT_PATH.read_text())
        if (return_code == 0) != (base_result.get("positive") is True):
            raise RuntimeError("base_result_return_mismatch")
        gc.collect()
        process_memory = r1.psutil.Process().memory_info()
        final_available = int(r1.psutil.virtual_memory().available)
        peak = int(getattr(process_memory, "peak_wset", process_memory.rss))
        if final_available < r1.RESERVE_MIN or peak > r1.PEAK_MAX:
            raise RuntimeError("blocked_resource_after_compute")
        handoff = {
            "kind": "het_next_l0_ph1_cpu_freeze_r2_handoff",
            "base_status": base_result["status"],
            "base_positive": base_result["positive"],
            "return_code": return_code,
            "authorization": authorization,
            "bindings": bindings,
            "runner_sha256": r1.sha_file(Path(__file__)),
            "r2_prereg_sha256": r1.sha_file(R2_PREREG),
            "resource": {
                **resource_start,
                "final_available": final_available,
                "peak_wset": peak,
                "reserve_min": r1.RESERVE_MIN,
                "peak_max": r1.PEAK_MAX,
            },
            "prepayload_quarantine": quarantine,
            "promotion_method": "MoveFileExW_MOVEFILE_WRITE_THROUGH" if os.name == "nt" else "rename_plus_parent_fsync",
            "device_or_compiler_opened": False,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        }
        r1.atomic_new(package / "handoff.json", r1.canonical(handoff))
        data_names = sorted(path.name for path in package.iterdir() if path.is_file())
        files = [
            {"name": name, "bytes": (package / name).stat().st_size, "sha256": r1.sha_file(package / name)}
            for name in data_names
        ]
        manifest = {"kind": "ph1_cpu_freeze_r2_manifest", "files": files}
        r1.atomic_new(package / "manifest.json", r1.canonical(manifest))
        commit = {
            "kind": "ph1_cpu_freeze_r2_commit",
            "manifest_sha256": r1.sha_file(package / "manifest.json"),
            "handoff_sha256": r1.sha_file(package / "handoff.json"),
            "base_result_sha256": r1.sha_file(package / "cpu_stage_freeze.json"),
        }
        r1.atomic_new(package / "commit.json", r1.canonical(commit))
        if not exact_committed(package):
            raise RuntimeError("prepromotion_verification")
        durable_move(package, FINAL)
        if not exact_committed(FINAL):
            raise RuntimeError("postpromotion_verification")
        print(json.dumps({"status": base_result["status"], "positive": base_result["positive"], "package": str(FINAL)}, indent=2))
        return return_code
    except Exception as exc:
        original_error = exc
        original_traceback = traceback.format_exc()
        secondary = None
        try:
            if package.exists():
                failure = {
                    "kind": "het_next_l0_ph1_cpu_freeze_r2_failure",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": original_traceback,
                    "device_or_compiler_opened": False,
                    "completed_utc": datetime.now(timezone.utc).isoformat(),
                }
                r1.atomic_new(package / "failure.json", r1.canonical(failure))
                target = FAILED / (package.name + ".failed")
                durable_move(package, target)
        except Exception as evidence_error:
            secondary = f"{type(evidence_error).__name__}: {evidence_error}"
        if secondary:
            original_error.add_note("secondary_failure_evidence_error=" + secondary)
        raise original_error.with_traceback(exc.__traceback__)
    finally:
        base.save_file = original_save
        base.torch.set_num_interop_threads = original_interop


if __name__ == "__main__":
    raise SystemExit(main())
