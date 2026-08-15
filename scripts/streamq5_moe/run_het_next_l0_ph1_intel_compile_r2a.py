#!/usr/bin/env python3
"""Final R2A compile-only runner over frozen R2 transaction logic."""
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
import het_next_l0_ph1_intel_compile_r2a_backend as backend
import run_het_next_l0_ph1_intel_compile_r2 as frozen_runner

REPORTS = ROOT / "reports/streamq5_moe"
OUT = REPORTS / "het_next_l0_ph1_intel_compile_r2a"
FAILED = REPORTS / "het_next_l0_ph1_intel_compile_r2a_failed_attempts"
QUARANTINE = REPORTS / "het_next_l0_ph1_intel_compile_r2a_quarantine"
LOCK = REPORTS / "het_next_l0_ph1_intel_compile_r2a_lock.json"
AUTH = REPORTS / "HET_NEXT_L0_PH1_INTEL_COMPILE_R2A_AUTHORIZATION_2026-08-14.md"
AUTH_PREFLIGHT = SCRIPTS / "preflight_het_next_l0_ph1_intel_compile_r2a.py"
R2P1_PREFLIGHT = SCRIPTS / "preflight_het_next_l0_ph1_intel_compile_r2p1.py"
R2P1_REVISION = REPORTS / "HET_NEXT_L0_PH1_INTEL_COMPILE_R2P1_PREFLIGHT_REVISION_2026-08-14.md"
R2P1_LOCK = REPORTS / "het_next_l0_ph1_intel_compile_r2p1_lock.json"
R2P1_PASS = REPORTS / "het_next_l0_ph1_intel_compile_r2p1_static_preflight.json"
R2_SOURCE_MODULE = SCRIPTS / "het_next_l0_ph1_intel_compile_r2_source.py"
R2_BACKEND = SCRIPTS / "het_next_l0_ph1_intel_compile_r2_backend.py"
R2_RUNNER = SCRIPTS / "run_het_next_l0_ph1_intel_compile_r2.py"
R2_PREREG = REPORTS / "HET_NEXT_L0_PH1_INTEL_COMPILE_R2_PREREGISTRATION_2026-08-14.md"
R2_DESIGN = REPORTS / "HET_NEXT_L0_PH1_INTEL_COMPILE_R2_SOURCE_REVISION_2026-08-14.md"
R2_LOCK = REPORTS / "het_next_l0_ph1_intel_compile_r2_lock.json"
R1B_FAILURE = REPORTS / "het_next_l0_ph1_intel_compile_r1b_failed_attempts/attempt_failure_06df3c72c9c44379a04d39b43d301b53/failure.json"


def configure() -> None:
    frozen_runner.OUT, frozen_runner.FAILED, frozen_runner.QUARANTINE = OUT, FAILED, QUARANTINE
    frozen_runner.base.OUT, frozen_runner.base.FAILED, frozen_runner.base.QUARANTINE = OUT, FAILED, QUARANTINE
    frozen_runner.base.verify_bundle = verify_bundle


def verify_bundle(directory: Path) -> dict:
    base = frozen_runner.base
    result_path, manifest_path, commit_path = (directory / name for name in ("result.json", "manifest.json", "commit.json"))
    if not all(path.is_file() for path in (result_path, manifest_path, commit_path)):
        raise RuntimeError("bundle_core_missing")
    result, manifest, commit = (json.loads(path.read_text(encoding="utf-8")) for path in (result_path, manifest_path, commit_path))
    if not (result.get("kind") == "het_next_l0_ph1_intel_compile_r2a" and manifest.get("kind") == "het_next_l0_ph1_intel_compile_r2a_manifest" and commit.get("kind") == "het_next_l0_ph1_intel_compile_r2a_commit" and commit.get("result_sha256") == base.file_sha(result_path) and commit.get("manifest_sha256") == base.file_sha(manifest_path)):
        raise RuntimeError("bundle_core_contract")
    rows = manifest.get("files")
    if not isinstance(rows, list) or not rows or {row["name"] for row in rows} | {"manifest.json", "commit.json"} != {path.name for path in directory.iterdir() if path.is_file()}:
        raise RuntimeError("bundle_manifest")
    for row in rows:
        path = directory / row["name"]
        if path.stat().st_size != row["bytes"] or base.file_sha(path) != row["sha256"]:
            raise RuntimeError("bundle_file_hash")
    return {"result": result, "manifest": manifest, "commit": commit}


def authorization() -> tuple[dict, dict]:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    passed = json.loads(R2P1_PASS.read_text(encoding="utf-8"))
    base = frozen_runner.base
    observed = {
        "backend_sha256": base.file_sha(Path(backend.__file__)), "runner_sha256": base.file_sha(Path(__file__)),
        "authorization_sha256": base.file_sha(AUTH), "authorization_preflight_sha256": base.file_sha(AUTH_PREFLIGHT),
        "r2p1_preflight_sha256": base.file_sha(R2P1_PREFLIGHT), "r2p1_revision_sha256": base.file_sha(R2P1_REVISION),
        "r2p1_lock_sha256": base.file_sha(R2P1_LOCK), "r2p1_pass_sha256": base.file_sha(R2P1_PASS),
        "r2_source_module_sha256": base.file_sha(R2_SOURCE_MODULE), "r2_backend_sha256": base.file_sha(R2_BACKEND),
        "r2_runner_sha256": base.file_sha(R2_RUNNER), "r2_prereg_sha256": base.file_sha(R2_PREREG),
        "r2_design_sha256": base.file_sha(R2_DESIGN), "r2_closed_lock_sha256": base.file_sha(R2_LOCK),
        "r1b_failure_sha256": base.file_sha(R1B_FAILURE), "source_sha256": base.sha_bytes(backend.SRC.encode()),
        "cpu_commit_sha256": base.file_sha(frozen_runner.CPU_COMMIT), "cpu_verification_sha256": base.file_sha(frozen_runner.CPU_VERIFY),
        "prior_audit_sha256": base.file_sha(frozen_runner.R1_AUDIT),
    }
    if not (lock.get("kind") == "het_next_l0_ph1_intel_compile_r2a_lock" and lock.get("execution_open") is True and lock.get("audit_token") == backend.ACK and all(lock.get(k) == v for k, v in observed.items()) and passed.get("pass") is True and passed.get("passed") == passed.get("total") == 8 and passed.get("compiler_calls") == passed.get("device_calls") == passed.get("payload_reads") == 0):
        raise RuntimeError("r2a_lock")
    return {"lock_sha256": base.file_sha(LOCK), "observed": observed}, {"cpu_commit_sha256": observed["cpu_commit_sha256"], "cpu_verification_sha256": observed["cpu_verification_sha256"], "prior_audit_sha256": observed["prior_audit_sha256"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ack", required=True)
    args = parser.parse_args()
    if args.ack != backend.ACK:
        raise SystemExit("ack")
    configure()
    base = frozen_runner.base
    recovered = base.recover()
    if recovered["already_complete"]:
        print(json.dumps({"status": "already_complete", "output": str(OUT)}, indent=2))
        return 0
    bindings, eligibility = authorization()
    attempt = REPORTS / f"{OUT.name}.{uuid.uuid4().hex}.inprogress"
    attempt.mkdir()
    try:
        compiled = backend.compile_only(eligibility)
        # Reuse R2 binary gates/writes, then rewrite only R2A JSON kinds before commit.
        source = compiled.pop("source").encode(); binary = bytes.fromhex(compiled.pop("binary_hex")); log = bytes.fromhex(compiled.pop("build_log_hex"))
        if not (compiled.get("binary_nonempty") is True and compiled.get("queried_program_devices") == 1 and compiled.get("declared_binary_bytes") == compiled.get("read_binary_bytes") == len(binary) > 0 and compiled.get("binary_sha256") == base.sha_bytes(binary) and not compiled.get("cleanup_errors") and compiled.get("payload_read") is False and all(compiled.get(k) == 0 for k in ("queues_created", "kernels_created", "events_created", "memory_objects_created", "allocations", "kernels_launched"))):
            raise RuntimeError("compile_positive_gate")
        for name, data in (("intel_source.cl", source), ("intel_program.bin", binary), ("intel_build.log", log)):
            base.write_new(attempt / name, data)
        result = {"kind": "het_next_l0_ph1_intel_compile_r2a", "status": "compile_positive", "positive": True, "completed_utc": base.utc(), "bindings": bindings, "compile": compiled, "artifacts": {"source_sha256": base.sha_bytes(source), "binary_sha256": base.sha_bytes(binary), "binary_bytes": len(binary), "build_log_sha256": base.sha_bytes(log), "build_log_bytes": len(log)}}
        base.write_new(attempt / "result.json", base.canonical(result))
        rows = [{"name": p.name, "bytes": p.stat().st_size, "sha256": base.file_sha(p)} for p in sorted(attempt.iterdir())]
        base.write_new(attempt / "manifest.json", base.canonical({"kind": "het_next_l0_ph1_intel_compile_r2a_manifest", "files": rows}))
        base.write_new(attempt / "commit.json", base.canonical({"kind": "het_next_l0_ph1_intel_compile_r2a_commit", "result_sha256": base.file_sha(attempt / "result.json"), "manifest_sha256": base.file_sha(attempt / "manifest.json")}))
        verify_bundle(attempt)
        base.durable_move(attempt, OUT)
        verify_bundle(OUT)
    except Exception as exc:
        base.archive(FAILED, "attempt_failure", {"kind": "het_next_l0_ph1_intel_compile_r2a_failure", "utc": base.utc(), "error": f"{type(exc).__name__}:{exc}", "traceback": traceback.format_exc(), "backend_evidence": getattr(exc, "evidence", None)}, attempt if attempt.exists() else (OUT if OUT.exists() else None))
        return 3
    print(json.dumps({"status": result["status"], "positive": True, "output": str(OUT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
