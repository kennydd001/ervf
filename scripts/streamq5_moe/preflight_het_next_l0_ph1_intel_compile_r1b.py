#!/usr/bin/env python3
"""R1B authorization preflight: AST-only inspection, no compiler/device/payload call."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts/streamq5_moe"
REPORTS = ROOT / "reports/streamq5_moe"
BACKEND = SCRIPTS / "het_next_l0_ph1_intel_compile_r1b_backend.py"
RUNNER = SCRIPTS / "run_het_next_l0_ph1_intel_compile_r1b.py"
AUTH = REPORTS / "HET_NEXT_L0_PH1_INTEL_COMPILE_R1B_AUTHORIZATION_2026-08-14.md"
LOCK = REPORTS / "het_next_l0_ph1_intel_compile_r1b_lock.json"
R1A_BACKEND = SCRIPTS / "het_next_l0_ph1_intel_compile_r1a_backend.py"
R1A_RUNNER = SCRIPTS / "run_het_next_l0_ph1_intel_compile_r1a.py"
R1A_PREFLIGHT = SCRIPTS / "preflight_het_next_l0_ph1_intel_compile_r1a.py"
R1A_AUTH = REPORTS / "HET_NEXT_L0_PH1_INTEL_COMPILE_R1A_AUTHORIZATION_2026-08-14.md"
R1A_LOCK = REPORTS / "het_next_l0_ph1_intel_compile_r1a_lock.json"
R1_BACKEND = SCRIPTS / "het_next_l0_ph1_intel_compile_r1_backend.py"
R1_RUNNER = SCRIPTS / "run_het_next_l0_ph1_intel_compile_r1.py"
R1_PREFLIGHT = SCRIPTS / "preflight_het_next_l0_ph1_intel_compile_r1.py"
R1_PREREG = REPORTS / "HET_NEXT_L0_PH1_INTEL_COMPILE_R1_PREREGISTRATION_2026-08-14.md"
R1_LOCK = REPORTS / "het_next_l0_ph1_intel_compile_r1_lock.json"
R1_PASS = REPORTS / "het_next_l0_ph1_intel_compile_r1_preflight_result.json"
AUDIT = REPORTS / "HET_NEXT_L0_PH1_INTEL_COMPILE_R0_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md"
OUT = REPORTS / "het_next_l0_ph1_intel_compile_r1b"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module_import_roots(tree: ast.AST) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def call_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                names.add(node.func.attr)
    return names


def static_surface_safe() -> bool:
    own_tree = ast.parse(Path(__file__).read_text(encoding="utf-8"), filename=str(Path(__file__)))
    backend_tree = ast.parse(BACKEND.read_text(encoding="utf-8"), filename=str(BACKEND))
    runner_tree = ast.parse(RUNNER.read_text(encoding="utf-8"), filename=str(RUNNER))
    forbidden_imports = {"ctypes", "torch", "cupy", "safetensors", "transformers", "mmap", "subprocess"}
    forbidden_calls = {
        "WinDLL",
        "CDLL",
        "Popen",
        "run",
        "check_call",
        "check_output",
        "compile_only",
        "safe_open",
        "from_pretrained",
    }
    return (
        not (module_import_roots(own_tree) & forbidden_imports)
        and not (call_names(own_tree) & forbidden_calls)
        and isinstance(backend_tree, ast.Module)
        and isinstance(runner_tree, ast.Module)
    )


def main() -> int:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    passed = json.loads(R1_PASS.read_text(encoding="utf-8"))
    observed = {
        "backend_sha256": sha(BACKEND),
        "runner_sha256": sha(RUNNER),
        "authorization_preflight_sha256": sha(Path(__file__)),
        "authorization_sha256": sha(AUTH),
        "r1a_backend_sha256": sha(R1A_BACKEND),
        "r1a_runner_sha256": sha(R1A_RUNNER),
        "r1a_preflight_sha256": sha(R1A_PREFLIGHT),
        "r1a_authorization_sha256": sha(R1A_AUTH),
        "r1a_lock_sha256": sha(R1A_LOCK),
        "r1_preflight_pass_sha256": sha(R1_PASS),
        "prior_audit_sha256": sha(AUDIT),
    }
    tests = {
        "self_hash_and_complete_chain": all(lock.get(key) == value for key, value in observed.items()),
        "open_exact_token": lock.get("execution_open") is True and lock.get("audit_token") == "PH1_INTEL_COMPILE_R1B_AFTER_PREFLIGHT_PASS_AND_INDEPENDENT_FINAL_AUDIT_GO",
        "closed_r1_pass_exact": passed.get("pass") is True and passed.get("passed") == passed.get("total") == 8,
        "closed_r1_zero_calls": passed.get("device_calls") == passed.get("compiler_calls") == passed.get("payload_reads") == 0,
        "closed_r1_self_bound": passed.get("preflight_sha256") == sha(R1_PREFLIGHT) and passed.get("source_lock_sha256") == sha(R1_LOCK),
        "frozen_r1_hashes": lock.get("r1_backend_sha256") == sha(R1_BACKEND) and lock.get("r1_runner_sha256") == sha(R1_RUNNER) and lock.get("r1_preflight_sha256") == sha(R1_PREFLIGHT) and lock.get("r1_prereg_sha256") == sha(R1_PREREG) and lock.get("r1_closed_lock_sha256") == sha(R1_LOCK),
        "ast_only_preflight_surface": static_surface_safe(),
        "output_absent": not OUT.exists(),
    }
    result = {"kind": "het_next_l0_ph1_intel_compile_r1b_authorization_preflight", "tests": tests, "pass": all(tests.values()), "passed": sum(tests.values()), "total": len(tests), "compiler_calls": 0, "device_calls": 0, "payload_reads": 0}
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0 if result["pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
