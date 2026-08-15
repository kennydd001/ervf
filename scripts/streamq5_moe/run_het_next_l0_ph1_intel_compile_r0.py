#!/usr/bin/env python3
"""Compile-only PH1 Intel source capture. No payload or kernel path."""
from __future__ import annotations

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
import het_next_l0_ph1_intel_backend as backend

REPORTS = ROOT / "reports/streamq5_moe"
OUT = REPORTS / "het_next_l0_ph1_intel_compile_r0"
FAILED = REPORTS / "het_next_l0_ph1_intel_compile_r0_failed_attempts"
PREREG = REPORTS / "HET_NEXT_L0_PH1_INTEL_COMPILE_R0_PREREGISTRATION_2026-08-14.md"
LOCK = REPORTS / "het_next_l0_ph1_intel_compile_lock.json"
CPU_COMMIT = REPORTS / "het_next_l0_ph1_cpu_freeze_r2/commit.json"
CPU_VERIFY = REPORTS / "het_next_l0_ph1_cpu_freeze_r2_independent_verification.json"
PHYSICAL = REPORTS / "HET_NEXT_L0_PH1_R1_CPU_EVIDENCE_AND_PHYSICAL_CONTRACT_2026-08-14.md"
CONTEXT = REPORTS / "HET_NEXT_L0_PH1_R2_NVIDIA_CONTEXT_CONTRACT_2026-08-14.md"
ACK = "PH1_INTEL_COMPILE_AFTER_SOURCE_AND_PREFLIGHT_GO"
EXPECTED = {
    CPU_COMMIT: "f3677e9610bea03649fec172b97c0c314f2f2e4c0d40bf9d864df0ec88a44f06",
    CPU_VERIFY: "1c7f2772fb637485020be00f74b6f9295a18ec3d7d10af0587ea350e8756cbc8",
    PHYSICAL: "7097a304eb6cd082367472cbc4c84ff9792414f3dd67e2590ba55b61dac3e981",
    CONTEXT: "dde29c369c5218f5cca3ed12248979a8c03c95b51e8b433f65175750d74d695c",
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode() + b"\n"


def write(path: Path, data: bytes) -> None:
    if path.exists():
        raise FileExistsError(path)
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def move(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    if os.name == "nt":
        import ctypes
        function = ctypes.windll.kernel32.MoveFileExW
        function.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32]
        function.restype = ctypes.c_int
        if not function(str(source), str(destination), 0x8):
            raise ctypes.WinError(ctypes.get_last_error())
    else:
        os.rename(source, destination)


def gate() -> tuple[dict, dict]:
    lock = json.loads(LOCK.read_text())
    if not (
        lock.get("kind") == "het_next_l0_ph1_intel_compile_lock"
        and lock.get("execution_open") is True
        and lock.get("audit_token") == ACK
        and lock.get("runner_sha256") == file_sha(Path(__file__))
        and lock.get("backend_sha256") == file_sha(Path(backend.__file__))
        and lock.get("prereg_sha256") == file_sha(PREREG)
        and lock.get("source_sha256") == sha(backend.SRC.encode())
    ):
        raise RuntimeError("compile_lock")
    observed = {str(path.relative_to(ROOT)): file_sha(path) for path in EXPECTED}
    if any(observed[str(path.relative_to(ROOT))] != expected for path, expected in EXPECTED.items()):
        raise RuntimeError("eligibility_hash")
    cpu_verification = json.loads(CPU_VERIFY.read_text())
    if cpu_verification.get("pass") is not True:
        raise RuntimeError("cpu_verification")
    eligibility = {
        "cpu_commit_sha256": EXPECTED[CPU_COMMIT],
        "cpu_verification_sha256": EXPECTED[CPU_VERIFY],
        "source_sha256": sha(backend.SRC.encode()),
    }
    return {"lock_sha256": file_sha(LOCK), "observed": observed}, eligibility


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ack", required=True)
    args = parser.parse_args()
    if args.ack != ACK:
        raise SystemExit("ack")
    if OUT.exists():
        raise FileExistsError(OUT)
    bindings, eligibility = gate()
    attempt = REPORTS / (OUT.name + "." + uuid.uuid4().hex + ".inprogress")
    attempt.mkdir()
    result = {
        "kind": "het_next_l0_ph1_intel_compile_r0",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "bindings": bindings,
        "runner_sha256": file_sha(Path(__file__)),
        "backend_sha256": file_sha(Path(backend.__file__)),
        "prereg_sha256": file_sha(PREREG),
    }
    try:
        compiled = backend.compile_only(eligibility)
        source = compiled.pop("source").encode()
        binary = bytes.fromhex(compiled.pop("binary_hex"))
        log = bytes.fromhex(compiled.pop("build_log_hex"))
        write(attempt / "intel_source.cl", source)
        write(attempt / "intel_program.bin", binary)
        write(attempt / "intel_build.log", log)
        result.update({"status": "compile_positive", "positive": True, "compile": compiled, "artifacts": {"source_sha256": sha(source), "binary_sha256": sha(binary), "binary_bytes": len(binary), "build_log_sha256": sha(log), "build_log_bytes": len(log)}})
    except backend.IntelRunFailure as exc:
        result.update({"status": "compile_failure", "positive": False, "error": str(exc), "compile": exc.evidence, "traceback": traceback.format_exc()})
    except Exception as exc:
        result.update({"status": "compile_failure", "positive": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
    result["completed_utc"] = datetime.now(timezone.utc).isoformat()
    write(attempt / "result.json", canonical(result))
    files = [{"name": p.name, "bytes": p.stat().st_size, "sha256": file_sha(p)} for p in sorted(attempt.iterdir())]
    write(attempt / "manifest.json", canonical({"kind": "ph1_intel_compile_r0_manifest", "files": files}))
    write(attempt / "commit.json", canonical({"kind": "ph1_intel_compile_r0_commit", "manifest_sha256": file_sha(attempt / "manifest.json"), "result_sha256": file_sha(attempt / "result.json")}))
    move(attempt, OUT)
    print(json.dumps({"status": result["status"], "positive": result["positive"], "output": str(OUT)}, indent=2))
    return 0 if result["positive"] is True else 3


if __name__ == "__main__":
    raise SystemExit(main())
