#!/usr/bin/env python3
"""Static final R2A authorization check; no imports of runtime modules or device calls."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts/streamq5_moe"
REPORTS = ROOT / "reports/streamq5_moe"
SELF = Path(__file__)
BACKEND = SCRIPTS / "het_next_l0_ph1_intel_compile_r2a_backend.py"
RUNNER = SCRIPTS / "run_het_next_l0_ph1_intel_compile_r2a.py"
AUTH = REPORTS / "HET_NEXT_L0_PH1_INTEL_COMPILE_R2A_AUTHORIZATION_2026-08-14.md"
LOCK = REPORTS / "het_next_l0_ph1_intel_compile_r2a_lock.json"
P1_PREFLIGHT = SCRIPTS / "preflight_het_next_l0_ph1_intel_compile_r2p1.py"
P1_REVISION = REPORTS / "HET_NEXT_L0_PH1_INTEL_COMPILE_R2P1_PREFLIGHT_REVISION_2026-08-14.md"
P1_LOCK = REPORTS / "het_next_l0_ph1_intel_compile_r2p1_lock.json"
P1_PASS = REPORTS / "het_next_l0_ph1_intel_compile_r2p1_static_preflight.json"
R2_SOURCE = SCRIPTS / "het_next_l0_ph1_intel_compile_r2_source.py"
R2_BACKEND = SCRIPTS / "het_next_l0_ph1_intel_compile_r2_backend.py"
R2_RUNNER = SCRIPTS / "run_het_next_l0_ph1_intel_compile_r2.py"
R2_PREREG = REPORTS / "HET_NEXT_L0_PH1_INTEL_COMPILE_R2_PREREGISTRATION_2026-08-14.md"
R2_DESIGN = REPORTS / "HET_NEXT_L0_PH1_INTEL_COMPILE_R2_SOURCE_REVISION_2026-08-14.md"
R2_LOCK = REPORTS / "het_next_l0_ph1_intel_compile_r2_lock.json"
R1B_FAILURE = REPORTS / "het_next_l0_ph1_intel_compile_r1b_failed_attempts/attempt_failure_06df3c72c9c44379a04d39b43d301b53/failure.json"
OUT = REPORTS / "het_next_l0_ph1_intel_compile_r2a"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    lock = json.loads(LOCK.read_text(encoding="utf-8")); passed = json.loads(P1_PASS.read_text(encoding="utf-8"))
    observed = {"backend_sha256": sha(BACKEND), "runner_sha256": sha(RUNNER), "authorization_sha256": sha(AUTH), "authorization_preflight_sha256": sha(SELF), "r2p1_preflight_sha256": sha(P1_PREFLIGHT), "r2p1_revision_sha256": sha(P1_REVISION), "r2p1_lock_sha256": sha(P1_LOCK), "r2p1_pass_sha256": sha(P1_PASS), "r2_source_module_sha256": sha(R2_SOURCE), "r2_backend_sha256": sha(R2_BACKEND), "r2_runner_sha256": sha(R2_RUNNER), "r2_prereg_sha256": sha(R2_PREREG), "r2_design_sha256": sha(R2_DESIGN), "r2_closed_lock_sha256": sha(R2_LOCK), "r1b_failure_sha256": sha(R1B_FAILURE)}
    trees = [ast.parse(path.read_text(encoding="utf-8"), filename=str(path)) for path in (SELF, BACKEND, RUNNER)]
    own_imports = set()
    for node in ast.walk(trees[0]):
        if isinstance(node, ast.Import): own_imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module: own_imports.add(node.module.split(".")[0])
    tests = {
        "open_exact_token_and_self_bound": lock.get("execution_open") is True and lock.get("audit_token") == "PH1_INTEL_COMPILE_R2A_AFTER_R2P1_PASS_AND_INDEPENDENT_FINAL_AUDIT_GO" and all(lock.get(k) == v for k, v in observed.items()),
        "r2p1_exact_pass": passed.get("pass") is True and passed.get("passed") == passed.get("total") == 8,
        "r2p1_zero_calls": passed.get("compiler_calls") == passed.get("device_calls") == passed.get("payload_reads") == 0,
        "source_unchanged": lock.get("source_sha256") == "f1b3ccdae6d202ed210810e3cd419f726ea89ffa8fba0c84df5c2bfca3a84d21",
        "static_preflight_surface": not (own_imports & {"ctypes", "torch", "cupy", "safetensors", "subprocess", "mmap"}),
        "runner_transaction_contract": all(name in {node.name for node in trees[2].body if isinstance(node, ast.FunctionDef)} for name in ("configure", "verify_bundle", "authorization", "main")),
        "output_absent": not OUT.exists(),
    }
    result = {"kind": "het_next_l0_ph1_intel_compile_r2a_authorization_preflight", "tests": tests, "pass": all(tests.values()), "passed": sum(tests.values()), "total": len(tests), "compiler_calls": 0, "device_calls": 0, "payload_reads": 0}
    print(json.dumps(result, sort_keys=True, indent=2)); return 0 if result["pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
