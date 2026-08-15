#!/usr/bin/env python3
"""One-shot PH1-R1 Intel compile-only capture with recoverable create-new artifacts."""
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
import het_next_l0_ph1_intel_compile_r1_backend as backend

REPORTS = ROOT / "reports/streamq5_moe"
OUT = REPORTS / "het_next_l0_ph1_intel_compile_r1"
FAILED = REPORTS / "het_next_l0_ph1_intel_compile_r1_failed_attempts"
QUARANTINE = REPORTS / "het_next_l0_ph1_intel_compile_r1_quarantine"
PREREG = REPORTS / "HET_NEXT_L0_PH1_INTEL_COMPILE_R1_PREREGISTRATION_2026-08-14.md"
AUDIT = REPORTS / "HET_NEXT_L0_PH1_INTEL_COMPILE_R0_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md"
LOCK = REPORTS / "het_next_l0_ph1_intel_compile_r1_lock.json"
PREFLIGHT = SCRIPTS / "preflight_het_next_l0_ph1_intel_compile_r1.py"
CPU_COMMIT = REPORTS / "het_next_l0_ph1_cpu_freeze_r2/commit.json"
CPU_VERIFY = REPORTS / "het_next_l0_ph1_cpu_freeze_r2_independent_verification.json"
PHYSICAL = REPORTS / "HET_NEXT_L0_PH1_R1_CPU_EVIDENCE_AND_PHYSICAL_CONTRACT_2026-08-14.md"
CONTEXT = REPORTS / "HET_NEXT_L0_PH1_R2_NVIDIA_CONTEXT_CONTRACT_2026-08-14.md"
ACK = backend.ACK
EXPECTED = {
    AUDIT: "ad1151b2a0a907e99ab0a99a6ac1b426587a14549fc4282821966f912544a841",
    CPU_COMMIT: "f3677e9610bea03649fec172b97c0c314f2f2e4c0d40bf9d864df0ec88a44f06",
    CPU_VERIFY: "1c7f2772fb637485020be00f74b6f9295a18ec3d7d10af0587ea350e8756cbc8",
    PHYSICAL: "7097a304eb6cd082367472cbc4c84ff9792414f3dd67e2590ba55b61dac3e981",
    CONTEXT: "dde29c369c5218f5cca3ed12248979a8c03c95b51e8b433f65175750d74d695c",
}


class RecoveryAbort(RuntimeError):
    pass


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
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
    commit_path = directory / "commit.json"
    manifest_path = directory / "manifest.json"
    result_path = directory / "result.json"
    if not all(p.is_file() for p in (commit_path, manifest_path, result_path)):
        raise RuntimeError("bundle_core_missing")
    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if (
        commit.get("kind") != "het_next_l0_ph1_intel_compile_r1_commit"
        or manifest.get("kind") != "het_next_l0_ph1_intel_compile_r1_manifest"
        or commit.get("manifest_sha256") != file_sha256(manifest_path)
        or commit.get("result_sha256") != file_sha256(result_path)
    ):
        raise RuntimeError("bundle_core_hash")
    declared = manifest.get("files")
    if not isinstance(declared, list) or not declared:
        raise RuntimeError("bundle_manifest_empty")
    expected_names = {row.get("name") for row in declared} | {"manifest.json", "commit.json"}
    actual_names = {p.name for p in directory.iterdir() if p.is_file()}
    if expected_names != actual_names:
        raise RuntimeError("bundle_file_set")
    for row in declared:
        path = directory / row["name"]
        if not path.is_file() or path.stat().st_size != row["bytes"] or file_sha256(path) != row["sha256"]:
            raise RuntimeError("bundle_file_hash:" + str(row.get("name")))
    return {"commit": commit, "manifest": manifest, "result": result}


def immutable_failure(root: Path, kind: str, payload: dict, attempt: Path | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"{kind}_{uuid.uuid4().hex}"
    if attempt is not None and attempt.exists():
        durable_move(attempt, destination)
    else:
        destination.mkdir()
    write_new(destination / "failure.json", canonical(payload))
    return destination


def recover_before_device(reports: Path = REPORTS, out: Path = OUT, failed: Path = FAILED, quarantine: Path = QUARANTINE) -> dict:
    """Return already_complete, or quarantine corrupt/stale state and abort this invocation."""
    if out.exists():
        try:
            verified = verify_bundle(out)
        except Exception as exc:
            destination = immutable_failure(
                quarantine,
                "corrupt_final",
                {"kind": "ph1_r1_recovery", "utc": utc(), "source": str(out), "error": f"{type(exc).__name__}:{exc}", "device_opened": False},
                out,
            )
            raise RecoveryAbort(f"corrupt_final_quarantined:{destination}") from exc
        return {"already_complete": True, "result": verified["result"], "path": str(out)}
    stale = sorted(reports.glob(out.name + ".*.inprogress"))
    if stale:
        moved = []
        for path in stale:
            destination = immutable_failure(
                quarantine,
                "stale_temp",
                {"kind": "ph1_r1_recovery", "utc": utc(), "source": str(path), "device_opened": False},
                path,
            )
            moved.append(str(destination))
        raise RecoveryAbort("stale_temp_quarantined:" + json.dumps(moved))
    return {"already_complete": False}


def gate() -> tuple[dict, dict]:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    exact = {
        "runner_sha256": file_sha256(Path(__file__)),
        "backend_sha256": file_sha256(Path(backend.__file__)),
        "preflight_sha256": file_sha256(PREFLIGHT),
        "prereg_sha256": file_sha256(PREREG),
        "source_sha256": sha256_bytes(backend.SRC.encode()),
        "prior_audit_sha256": file_sha256(AUDIT),
    }
    if not (
        lock.get("kind") == "het_next_l0_ph1_intel_compile_r1_lock"
        and lock.get("execution_open") is True
        and lock.get("audit_token") == ACK
        and all(lock.get(key) == value for key, value in exact.items())
        and lock.get("cpu_commit_sha256") == EXPECTED[CPU_COMMIT]
        and lock.get("cpu_verification_sha256") == EXPECTED[CPU_VERIFY]
        and lock.get("physical_contract_sha256") == EXPECTED[PHYSICAL]
        and lock.get("nvidia_context_contract_sha256") == EXPECTED[CONTEXT]
    ):
        raise RuntimeError("compile_r1_lock")
    observed = {str(path.relative_to(ROOT)): file_sha256(path) for path in EXPECTED}
    if any(observed[str(path.relative_to(ROOT))] != expected for path, expected in EXPECTED.items()):
        raise RuntimeError("eligibility_hash")
    if json.loads(CPU_VERIFY.read_text(encoding="utf-8")).get("pass") is not True:
        raise RuntimeError("cpu_verification")
    eligibility = {
        "cpu_commit_sha256": EXPECTED[CPU_COMMIT],
        "cpu_verification_sha256": EXPECTED[CPU_VERIFY],
        "prior_audit_sha256": EXPECTED[AUDIT],
    }
    return {"lock_sha256": file_sha256(LOCK), "exact": exact, "observed": observed}, eligibility


def build_positive_bundle(attempt: Path, bindings: dict, compiled: dict) -> dict:
    source = compiled.pop("source").encode()
    binary = bytes.fromhex(compiled.pop("binary_hex"))
    build_log = bytes.fromhex(compiled.pop("build_log_hex"))
    if not (
        compiled.get("binary_nonempty") is True
        and compiled.get("queried_program_devices") == 1
        and compiled.get("declared_binary_bytes") == compiled.get("read_binary_bytes") == len(binary)
        and len(binary) > 0
        and compiled.get("binary_sha256") == sha256_bytes(binary)
        and not compiled.get("cleanup_errors")
        and compiled.get("payload_read") is False
        and all(compiled.get(key) == 0 for key in ("queues_created", "kernels_created", "events_created", "memory_objects_created", "allocations", "kernels_launched"))
    ):
        raise RuntimeError("compile_positive_gate")
    write_new(attempt / "intel_source.cl", source)
    write_new(attempt / "intel_program.bin", binary)
    write_new(attempt / "intel_build.log", build_log)
    result = {
        "kind": "het_next_l0_ph1_intel_compile_r1",
        "status": "compile_positive",
        "positive": True,
        "started_utc": bindings.pop("started_utc"),
        "completed_utc": utc(),
        "bindings": bindings,
        "compile": compiled,
        "artifacts": {
            "source_sha256": sha256_bytes(source),
            "source_bytes": len(source),
            "binary_sha256": sha256_bytes(binary),
            "binary_bytes": len(binary),
            "build_log_sha256": sha256_bytes(build_log),
            "build_log_bytes": len(build_log),
        },
    }
    write_new(attempt / "result.json", canonical(result))
    files = [
        {"name": p.name, "bytes": p.stat().st_size, "sha256": file_sha256(p)}
        for p in sorted(attempt.iterdir())
        if p.is_file()
    ]
    write_new(attempt / "manifest.json", canonical({"kind": "het_next_l0_ph1_intel_compile_r1_manifest", "files": files}))
    write_new(
        attempt / "commit.json",
        canonical(
            {
                "kind": "het_next_l0_ph1_intel_compile_r1_commit",
                "manifest_sha256": file_sha256(attempt / "manifest.json"),
                "result_sha256": file_sha256(attempt / "result.json"),
            }
        ),
    )
    verify_bundle(attempt)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ack", required=True)
    args = parser.parse_args()
    if args.ack != ACK:
        raise SystemExit("ack")
    recovery = recover_before_device()
    if recovery["already_complete"]:
        print(json.dumps({"status": "already_complete", "output": str(OUT)}, indent=2))
        return 0
    bindings, eligibility = gate()
    bindings["started_utc"] = utc()
    attempt = REPORTS / f"{OUT.name}.{uuid.uuid4().hex}.inprogress"
    attempt.mkdir()
    device_opened = False
    try:
        device_opened = True
        compiled = backend.compile_only(eligibility)
        result = build_positive_bundle(attempt, bindings, compiled)
        durable_move(attempt, OUT)
        verify_bundle(OUT)
    except Exception as exc:
        evidence = getattr(exc, "evidence", None)
        payload = {
            "kind": "het_next_l0_ph1_intel_compile_r1_failure",
            "utc": utc(),
            "device_opened": device_opened,
            "error": f"{type(exc).__name__}:{exc}",
            "traceback": traceback.format_exc(),
            "backend_evidence": evidence,
            "attempt_exists": attempt.exists(),
            "final_exists": OUT.exists(),
        }
        disposition = attempt if attempt.exists() else (OUT if OUT.exists() else None)
        immutable_failure(FAILED, "attempt_failure", payload, disposition)
        return 3
    print(json.dumps({"status": result["status"], "positive": True, "output": str(OUT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
