#!/usr/bin/env python3
"""Authorized R1A one-shot Intel compile-only runner; no science change from R1."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts/streamq5_moe"
sys.path.insert(0, str(SCRIPTS))
import het_next_l0_ph1_intel_compile_r1a_backend as backend

REPORTS = ROOT / "reports/streamq5_moe"
OUT = REPORTS / "het_next_l0_ph1_intel_compile_r1a"
FAILED = REPORTS / "het_next_l0_ph1_intel_compile_r1a_failed_attempts"
QUARANTINE = REPORTS / "het_next_l0_ph1_intel_compile_r1a_quarantine"
LOCK = REPORTS / "het_next_l0_ph1_intel_compile_r1a_lock.json"
AUTH = REPORTS / "HET_NEXT_L0_PH1_INTEL_COMPILE_R1A_AUTHORIZATION_2026-08-14.md"
R1_PREREG = REPORTS / "HET_NEXT_L0_PH1_INTEL_COMPILE_R1_PREREGISTRATION_2026-08-14.md"
R1_AUDIT = REPORTS / "HET_NEXT_L0_PH1_INTEL_COMPILE_R0_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md"
R1_LOCK = REPORTS / "het_next_l0_ph1_intel_compile_r1_lock.json"
PASS_RESULT = REPORTS / "het_next_l0_ph1_intel_compile_r1_preflight_result.json"
R1_PREFLIGHT = SCRIPTS / "preflight_het_next_l0_ph1_intel_compile_r1.py"
R1_RUNNER = SCRIPTS / "run_het_next_l0_ph1_intel_compile_r1.py"
R1A_PREFLIGHT = SCRIPTS / "preflight_het_next_l0_ph1_intel_compile_r1a.py"
CPU_COMMIT = REPORTS / "het_next_l0_ph1_cpu_freeze_r2/commit.json"
CPU_VERIFY = REPORTS / "het_next_l0_ph1_cpu_freeze_r2_independent_verification.json"
ACK = backend.ACK


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode() + b"\n"


def write_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def durable_move(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        import ctypes as C

        move = C.WinDLL("kernel32", use_last_error=True).MoveFileExW
        move.argtypes = [C.c_wchar_p, C.c_wchar_p, C.c_uint32]
        move.restype = C.c_int
        if not move(str(source), str(destination), 0x8):
            raise C.WinError(C.get_last_error())
    else:
        os.rename(source, destination)


def verify_bundle(directory: Path) -> dict:
    core = [directory / name for name in ("result.json", "manifest.json", "commit.json")]
    if not all(path.is_file() for path in core):
        raise RuntimeError("bundle_core_missing")
    result, manifest, commit = (json.loads(path.read_text(encoding="utf-8")) for path in core)
    if not (
        result.get("kind") == "het_next_l0_ph1_intel_compile_r1a"
        and manifest.get("kind") == "het_next_l0_ph1_intel_compile_r1a_manifest"
        and commit.get("kind") == "het_next_l0_ph1_intel_compile_r1a_commit"
        and commit.get("result_sha256") == file_sha(core[0])
        and commit.get("manifest_sha256") == file_sha(core[1])
    ):
        raise RuntimeError("bundle_core_contract")
    rows = manifest.get("files")
    expected = {row["name"] for row in rows} | {"manifest.json", "commit.json"}
    if expected != {path.name for path in directory.iterdir() if path.is_file()}:
        raise RuntimeError("bundle_file_set")
    for row in rows:
        path = directory / row["name"]
        if path.stat().st_size != row["bytes"] or file_sha(path) != row["sha256"]:
            raise RuntimeError("bundle_file_hash")
    return {"result": result, "manifest": manifest, "commit": commit}


def archive(root: Path, prefix: str, payload: dict, source: Path | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"{prefix}_{uuid.uuid4().hex}"
    if source is not None and source.exists():
        durable_move(source, destination)
    else:
        destination.mkdir()
    write_new(destination / "failure.json", canonical(payload))
    return destination


def recover() -> dict:
    if OUT.exists():
        try:
            return {"already_complete": True, **verify_bundle(OUT)}
        except Exception as exc:
            archive(QUARANTINE, "corrupt_final", {"utc": utc(), "error": f"{type(exc).__name__}:{exc}", "device_opened": False}, OUT)
            raise RuntimeError("corrupt_final_quarantined") from exc
    stale = sorted(REPORTS.glob(OUT.name + ".*.inprogress"))
    if stale:
        for path in stale:
            archive(QUARANTINE, "stale_temp", {"utc": utc(), "source": str(path), "device_opened": False}, path)
        raise RuntimeError("stale_temp_quarantined")
    return {"already_complete": False}


def authorization() -> tuple[dict, dict]:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    pass_result = json.loads(PASS_RESULT.read_text(encoding="utf-8"))
    observed = {
        "backend_sha256": file_sha(Path(backend.__file__)),
        "runner_sha256": file_sha(Path(__file__)),
        "authorization_sha256": file_sha(AUTH),
        "authorization_preflight_sha256": file_sha(R1A_PREFLIGHT),
        "r1_backend_sha256": file_sha(Path(backend.frozen_r1.__file__)),
        "r1_runner_sha256": file_sha(R1_RUNNER),
        "r1_preflight_sha256": file_sha(R1_PREFLIGHT),
        "r1_prereg_sha256": file_sha(R1_PREREG),
        "r1_closed_lock_sha256": file_sha(R1_LOCK),
        "r1_preflight_pass_sha256": file_sha(PASS_RESULT),
        "prior_audit_sha256": file_sha(R1_AUDIT),
        "source_sha256": sha_bytes(backend.SRC.encode()),
        "cpu_commit_sha256": file_sha(CPU_COMMIT),
        "cpu_verification_sha256": file_sha(CPU_VERIFY),
    }
    if not (
        lock.get("kind") == "het_next_l0_ph1_intel_compile_r1a_lock"
        and lock.get("execution_open") is True
        and lock.get("audit_token") == ACK
        and all(lock.get(key) == value for key, value in observed.items())
        and pass_result.get("pass") is True
        and pass_result.get("passed") == pass_result.get("total") == 8
        and pass_result.get("device_calls") == pass_result.get("compiler_calls") == pass_result.get("payload_reads") == 0
    ):
        raise RuntimeError("r1a_lock")
    eligibility = {
        "cpu_commit_sha256": observed["cpu_commit_sha256"],
        "cpu_verification_sha256": observed["cpu_verification_sha256"],
        "prior_audit_sha256": observed["prior_audit_sha256"],
    }
    return {"lock_sha256": file_sha(LOCK), "observed": observed}, eligibility


def build(attempt: Path, bindings: dict, compiled: dict) -> dict:
    source = compiled.pop("source").encode()
    binary = bytes.fromhex(compiled.pop("binary_hex"))
    log = bytes.fromhex(compiled.pop("build_log_hex"))
    if not (
        compiled.get("binary_nonempty") is True
        and compiled.get("queried_program_devices") == 1
        and compiled.get("declared_binary_bytes") == compiled.get("read_binary_bytes") == len(binary) > 0
        and compiled.get("binary_sha256") == sha_bytes(binary)
        and not compiled.get("cleanup_errors")
        and compiled.get("payload_read") is False
        and all(compiled.get(key) == 0 for key in ("queues_created", "kernels_created", "events_created", "memory_objects_created", "allocations", "kernels_launched"))
    ):
        raise RuntimeError("compile_positive_gate")
    for name, data in (("intel_source.cl", source), ("intel_program.bin", binary), ("intel_build.log", log)):
        write_new(attempt / name, data)
    result = {
        "kind": "het_next_l0_ph1_intel_compile_r1a",
        "status": "compile_positive",
        "positive": True,
        "completed_utc": utc(),
        "bindings": bindings,
        "compile": compiled,
        "artifacts": {
            "source_sha256": sha_bytes(source),
            "binary_sha256": sha_bytes(binary),
            "binary_bytes": len(binary),
            "build_log_sha256": sha_bytes(log),
            "build_log_bytes": len(log),
        },
    }
    write_new(attempt / "result.json", canonical(result))
    rows = [{"name": path.name, "bytes": path.stat().st_size, "sha256": file_sha(path)} for path in sorted(attempt.iterdir())]
    write_new(attempt / "manifest.json", canonical({"kind": "het_next_l0_ph1_intel_compile_r1a_manifest", "files": rows}))
    write_new(attempt / "commit.json", canonical({"kind": "het_next_l0_ph1_intel_compile_r1a_commit", "result_sha256": file_sha(attempt / "result.json"), "manifest_sha256": file_sha(attempt / "manifest.json")}))
    verify_bundle(attempt)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ack", required=True)
    args = parser.parse_args()
    if args.ack != ACK:
        raise SystemExit("ack")
    recovered = recover()
    if recovered["already_complete"]:
        print(json.dumps({"status": "already_complete", "output": str(OUT)}, indent=2))
        return 0
    bindings, eligibility = authorization()
    attempt = REPORTS / f"{OUT.name}.{uuid.uuid4().hex}.inprogress"
    attempt.mkdir()
    try:
        result = build(attempt, bindings, backend.compile_only(eligibility))
        durable_move(attempt, OUT)
        verify_bundle(OUT)
    except Exception as exc:
        payload = {"kind": "het_next_l0_ph1_intel_compile_r1a_failure", "utc": utc(), "error": f"{type(exc).__name__}:{exc}", "traceback": traceback.format_exc(), "backend_evidence": getattr(exc, "evidence", None)}
        archive(FAILED, "attempt_failure", payload, attempt if attempt.exists() else (OUT if OUT.exists() else None))
        return 3
    print(json.dumps({"status": result["status"], "positive": True, "output": str(OUT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
