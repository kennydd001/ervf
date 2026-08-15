#!/usr/bin/env python3
"""R1A authorization-chain check only; no import, compiler, payload or device call."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts/streamq5_moe"
REPORTS = ROOT / "reports/streamq5_moe"
BACKEND = SCRIPTS / "het_next_l0_ph1_intel_compile_r1a_backend.py"
RUNNER = SCRIPTS / "run_het_next_l0_ph1_intel_compile_r1a.py"
AUTH = REPORTS / "HET_NEXT_L0_PH1_INTEL_COMPILE_R1A_AUTHORIZATION_2026-08-14.md"
LOCK = REPORTS / "het_next_l0_ph1_intel_compile_r1a_lock.json"
R1_BACKEND = SCRIPTS / "het_next_l0_ph1_intel_compile_r1_backend.py"
R1_RUNNER = SCRIPTS / "run_het_next_l0_ph1_intel_compile_r1.py"
R1_PREFLIGHT = SCRIPTS / "preflight_het_next_l0_ph1_intel_compile_r1.py"
R1_PREREG = REPORTS / "HET_NEXT_L0_PH1_INTEL_COMPILE_R1_PREREGISTRATION_2026-08-14.md"
R1_LOCK = REPORTS / "het_next_l0_ph1_intel_compile_r1_lock.json"
PASS_RESULT = REPORTS / "het_next_l0_ph1_intel_compile_r1_preflight_result.json"
AUDIT = REPORTS / "HET_NEXT_L0_PH1_INTEL_COMPILE_R0_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md"
OUT = REPORTS / "het_next_l0_ph1_intel_compile_r1a"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    passed = json.loads(PASS_RESULT.read_text(encoding="utf-8"))
    observed = {
        "backend_sha256": sha(BACKEND),
        "runner_sha256": sha(RUNNER),
        "authorization_sha256": sha(AUTH),
        "r1_backend_sha256": sha(R1_BACKEND),
        "r1_runner_sha256": sha(R1_RUNNER),
        "r1_preflight_sha256": sha(R1_PREFLIGHT),
        "r1_prereg_sha256": sha(R1_PREREG),
        "r1_closed_lock_sha256": sha(R1_LOCK),
        "r1_preflight_pass_sha256": sha(PASS_RESULT),
        "prior_audit_sha256": sha(AUDIT),
    }
    backend_tree = ast.parse(BACKEND.read_text(encoding="utf-8"), filename=str(BACKEND))
    runner_tree = ast.parse(RUNNER.read_text(encoding="utf-8"), filename=str(RUNNER))
    forbidden = ("OpenCL.dll", "clBuildProgram", "WinDLL", "safe_open", "from_pretrained", ".safetensors")
    tests = {
        "open_exact_token": lock.get("execution_open") is True and lock.get("audit_token") == "PH1_INTEL_COMPILE_R1A_AFTER_PREFLIGHT_PASS_AND_INDEPENDENT_FINAL_AUDIT_GO",
        "authorization_chain": all(lock.get(key) == value for key, value in observed.items()),
        "closed_preflight_exact_pass": passed.get("pass") is True and passed.get("passed") == passed.get("total") == 8,
        "closed_preflight_zero_calls": passed.get("device_calls") == passed.get("compiler_calls") == passed.get("payload_reads") == 0,
        "closed_preflight_self_bound": passed.get("preflight_sha256") == observed["r1_preflight_sha256"] and passed.get("source_lock_sha256") == observed["r1_closed_lock_sha256"],
        "r1a_output_absent": not OUT.exists(),
        "syntax_static": isinstance(backend_tree, ast.Module) and isinstance(runner_tree, ast.Module),
        "this_preflight_has_no_device_surface": not any(token in Path(__file__).read_text(encoding="utf-8") for token in forbidden),
    }
    result = {"kind": "het_next_l0_ph1_intel_compile_r1a_authorization_preflight", "tests": tests, "pass": all(tests.values()), "passed": sum(tests.values()), "total": len(tests), "compiler_calls": 0, "device_calls": 0, "payload_reads": 0}
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0 if result["pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
