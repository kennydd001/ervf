#!/usr/bin/env python3
"""Closed PH1-R2 compile-only runner using frozen R1 transaction primitives."""
from __future__ import annotations

import argparse
import json
import sys
import traceback
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts/streamq5_moe"
sys.path.insert(0, str(SCRIPTS))
import het_next_l0_ph1_intel_compile_r2_backend as backend
import run_het_next_l0_ph1_intel_compile_r1a as base

REPORTS = ROOT / "reports/streamq5_moe"
OUT = REPORTS / "het_next_l0_ph1_intel_compile_r2"
FAILED = REPORTS / "het_next_l0_ph1_intel_compile_r2_failed_attempts"
QUARANTINE = REPORTS / "het_next_l0_ph1_intel_compile_r2_quarantine"
LOCK = REPORTS / "het_next_l0_ph1_intel_compile_r2_lock.json"
PREREG = REPORTS / "HET_NEXT_L0_PH1_INTEL_COMPILE_R2_PREREGISTRATION_2026-08-14.md"
DESIGN = REPORTS / "HET_NEXT_L0_PH1_INTEL_COMPILE_R2_SOURCE_REVISION_2026-08-14.md"
PREFLIGHT = SCRIPTS / "preflight_het_next_l0_ph1_intel_compile_r2.py"
SOURCE_MODULE = SCRIPTS / "het_next_l0_ph1_intel_compile_r2_source.py"
R1_BACKEND = SCRIPTS / "het_next_l0_ph1_intel_compile_r1_backend.py"
R1B_FAILURE = REPORTS / "het_next_l0_ph1_intel_compile_r1b_failed_attempts/attempt_failure_06df3c72c9c44379a04d39b43d301b53/failure.json"
R1_AUDIT = REPORTS / "HET_NEXT_L0_PH1_INTEL_COMPILE_R0_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md"
CPU_COMMIT = REPORTS / "het_next_l0_ph1_cpu_freeze_r2/commit.json"
CPU_VERIFY = REPORTS / "het_next_l0_ph1_cpu_freeze_r2_independent_verification.json"


def configure_base() -> None:
    base.OUT, base.FAILED, base.QUARANTINE = OUT, FAILED, QUARANTINE


def verify_bundle(directory: Path) -> dict:
    result_path, manifest_path, commit_path = (directory / name for name in ("result.json", "manifest.json", "commit.json"))
    if not all(path.is_file() for path in (result_path, manifest_path, commit_path)):
        raise RuntimeError("bundle_core_missing")
    result, manifest, commit = (json.loads(path.read_text(encoding="utf-8")) for path in (result_path, manifest_path, commit_path))
    if not (
        result.get("kind") == "het_next_l0_ph1_intel_compile_r2"
        and manifest.get("kind") == "het_next_l0_ph1_intel_compile_r2_manifest"
        and commit.get("kind") == "het_next_l0_ph1_intel_compile_r2_commit"
        and commit.get("result_sha256") == base.file_sha(result_path)
        and commit.get("manifest_sha256") == base.file_sha(manifest_path)
    ):
        raise RuntimeError("bundle_core_contract")
    rows = manifest.get("files")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("manifest_empty")
    if {row["name"] for row in rows} | {"manifest.json", "commit.json"} != {path.name for path in directory.iterdir() if path.is_file()}:
        raise RuntimeError("bundle_file_set")
    for row in rows:
        path = directory / row["name"]
        if path.stat().st_size != row["bytes"] or base.file_sha(path) != row["sha256"]:
            raise RuntimeError("bundle_file_hash")
    return {"result": result, "manifest": manifest, "commit": commit}


def authorization() -> tuple[dict, dict]:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    failure = json.loads(R1B_FAILURE.read_text(encoding="utf-8"))
    observed = {
        "backend_sha256": base.file_sha(Path(backend.__file__)),
        "runner_sha256": base.file_sha(Path(__file__)),
        "preflight_sha256": base.file_sha(PREFLIGHT),
        "prereg_sha256": base.file_sha(PREREG),
        "design_sha256": base.file_sha(DESIGN),
        "source_module_sha256": base.file_sha(SOURCE_MODULE),
        "r1_backend_sha256": base.file_sha(R1_BACKEND),
        "r1b_failure_sha256": base.file_sha(R1B_FAILURE),
        "prior_audit_sha256": base.file_sha(R1_AUDIT),
        "source_sha256": base.sha_bytes(backend.SRC.encode()),
        "cpu_commit_sha256": base.file_sha(CPU_COMMIT),
        "cpu_verification_sha256": base.file_sha(CPU_VERIFY),
    }
    cleanup = failure.get("backend_evidence", {}).get("ledger", [])[-1]
    zero = failure.get("backend_evidence", {})
    if not (
        lock.get("kind") == "het_next_l0_ph1_intel_compile_r2_lock"
        and lock.get("execution_open") is True
        and lock.get("audit_token") == backend.ACK
        and all(lock.get(key) == value for key, value in observed.items())
        and failure.get("backend_evidence", {}).get("error") == "RuntimeError:program_build:-11"
        and cleanup.get("cleanup_complete") is True
        and cleanup.get("live_owned_resources") == 0
        and all(zero.get(key) == 0 for key in ("queues_created", "kernels_created", "events_created", "memory_objects_created", "allocations", "kernels_launched"))
        and zero.get("payload_read") is False
    ):
        raise RuntimeError("r2_lock_or_failure_evidence")
    return {"lock_sha256": base.file_sha(LOCK), "observed": observed}, {"cpu_commit_sha256": observed["cpu_commit_sha256"], "cpu_verification_sha256": observed["cpu_verification_sha256"], "prior_audit_sha256": observed["prior_audit_sha256"]}


def build(attempt: Path, bindings: dict, compiled: dict) -> dict:
    source = compiled.pop("source").encode()
    binary = bytes.fromhex(compiled.pop("binary_hex"))
    log = bytes.fromhex(compiled.pop("build_log_hex"))
    if not (
        compiled.get("binary_nonempty") is True
        and compiled.get("queried_program_devices") == 1
        and compiled.get("declared_binary_bytes") == compiled.get("read_binary_bytes") == len(binary) > 0
        and compiled.get("binary_sha256") == base.sha_bytes(binary)
        and not compiled.get("cleanup_errors")
        and compiled.get("payload_read") is False
        and all(compiled.get(key) == 0 for key in ("queues_created", "kernels_created", "events_created", "memory_objects_created", "allocations", "kernels_launched"))
    ):
        raise RuntimeError("compile_positive_gate")
    for name, data in (("intel_source.cl", source), ("intel_program.bin", binary), ("intel_build.log", log)):
        base.write_new(attempt / name, data)
    result = {"kind": "het_next_l0_ph1_intel_compile_r2", "status": "compile_positive", "positive": True, "completed_utc": base.utc(), "bindings": bindings, "compile": compiled, "artifacts": {"source_sha256": base.sha_bytes(source), "binary_sha256": base.sha_bytes(binary), "binary_bytes": len(binary), "build_log_sha256": base.sha_bytes(log), "build_log_bytes": len(log)}}
    base.write_new(attempt / "result.json", base.canonical(result))
    rows = [{"name": path.name, "bytes": path.stat().st_size, "sha256": base.file_sha(path)} for path in sorted(attempt.iterdir())]
    base.write_new(attempt / "manifest.json", base.canonical({"kind": "het_next_l0_ph1_intel_compile_r2_manifest", "files": rows}))
    base.write_new(attempt / "commit.json", base.canonical({"kind": "het_next_l0_ph1_intel_compile_r2_commit", "result_sha256": base.file_sha(attempt / "result.json"), "manifest_sha256": base.file_sha(attempt / "manifest.json")}))
    verify_bundle(attempt)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ack", required=True)
    args = parser.parse_args()
    if args.ack != backend.ACK:
        raise SystemExit("ack")
    configure_base()
    base.verify_bundle = verify_bundle
    recovered = base.recover()
    if recovered["already_complete"]:
        print(json.dumps({"status": "already_complete", "output": str(OUT)}, indent=2))
        return 0
    bindings, eligibility = authorization()
    attempt = REPORTS / f"{OUT.name}.{uuid.uuid4().hex}.inprogress"
    attempt.mkdir()
    try:
        result = build(attempt, bindings, backend.compile_only(eligibility))
        base.durable_move(attempt, OUT)
        verify_bundle(OUT)
    except Exception as exc:
        base.archive(FAILED, "attempt_failure", {"kind": "het_next_l0_ph1_intel_compile_r2_failure", "utc": base.utc(), "error": f"{type(exc).__name__}:{exc}", "traceback": traceback.format_exc(), "backend_evidence": getattr(exc, "evidence", None)}, attempt if attempt.exists() else (OUT if OUT.exists() else None))
        return 3
    print(json.dumps({"status": result["status"], "positive": True, "output": str(OUT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
