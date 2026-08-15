#!/usr/bin/env python3
"""Closed no-device R7C2 preflight: device-state retention and full clean-state."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
S = ROOT / "scripts/streamq5_moe"
R = ROOT / "reports/streamq5_moe"
sys.path.insert(0, str(S))
import preflight_het_next_l0_ph1_intel_execution_r7c1 as prior_preflight
import run_het_next_l0_ph1_intel_execution_r7c2 as runner
import verify_het_next_l0_ph1_intel_execution_r7c2 as verifier

LOCK = R / "het_next_l0_ph1_intel_execution_r7c2_lock.json"
RESULT = R / "het_next_l0_ph1_intel_execution_r7c2_static_preflight.json"
R7P_RESULT = R / "het_next_l0_ph1_intel_execution_r7p_static_preflight.json"


def sha256(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()


def install_prior_fixture_modules() -> tuple:
    old = (prior_preflight.runner, prior_preflight.verifier, prior_preflight.LOCK, prior_preflight.RESULT)
    prior_preflight.runner, prior_preflight.verifier, prior_preflight.LOCK, prior_preflight.RESULT = runner, verifier, LOCK, RESULT
    return old


def restore_prior_fixture_modules(old: tuple) -> None:
    prior_preflight.runner, prior_preflight.verifier, prior_preflight.LOCK, prior_preflight.RESULT = old


def configure_case(root: Path) -> tuple:
    old = (runner.R, runner.OUTER_FAILED, runner.OUTER_QUARANTINE, runner.physical.OUT, runner.physical.FAILED)
    runner.R = root; runner.OUTER_FAILED = root / "r7c2_failed"; runner.OUTER_QUARANTINE = root / "r7c2_quarantine"
    runner.physical.OUT = root / "r7a_output"; runner.physical.FAILED = root / "r7a_failed"
    return old


def restore_case(old: tuple) -> None:
    runner.R, runner.OUTER_FAILED, runner.OUTER_QUARANTINE, runner.physical.OUT, runner.physical.FAILED = old


def device_opened_case(value) -> tuple[bool, dict]:
    with tempfile.TemporaryDirectory() as temporary:
        old = configure_case(Path(temporary))
        try:
            def executor(_):
                directory = runner.physical.FAILED / "attempt_device"; directory.mkdir(parents=True)
                (directory / "failure.json").write_text(json.dumps({"kind": "ph1_intel_execution_r7a_failure", "status": "valid_negative_failure", "error": "late", "device_opened": value, "disposition": "attempt_archived_create_new"}, sort_keys=True) + "\n")
                return 3
            rc = runner.outer_execute({}, executor); rows = list(runner.OUTER_FAILED.glob("*/failure.json")); summary = json.loads(rows[0].read_text()) if len(rows) == 1 else {}
            return rc == 3, summary
        finally: restore_case(old)


def extension_mutations() -> tuple[bool, list[str]]:
    observed = {name: sha256(path) for name, path in verifier.CHAIN.items()}; auth = json.loads(verifier.AUTH_RESULT.read_text())
    with tempfile.TemporaryDirectory() as temporary:
        old_lock = verifier.LOCK
        try:
            verifier.LOCK = Path(temporary) / "lock.json"; lock = json.loads(LOCK.read_text()); lock["execution_open"] = True; lock["audit_token"] = verifier.ACK; verifier.LOCK.write_text(json.dumps(lock, sort_keys=True, indent=2) + "\n")
            extension = {"lock_sha256": sha256(verifier.LOCK), "observed": observed, "authorization_result_sha256": verifier.AUTH_RESULT_SHA, "authorization_result": auth, "audit_token": verifier.ACK, "outer_failure_stages": verifier.OUTER_STAGES}; baseline = verifier.extension_valid(extension, lock, observed, auth); rejected = []
            mutations = {"token":lambda e,l,o,a:e.__setitem__("audit_token","bad"),"result_hash":lambda e,l,o,a:e.__setitem__("authorization_result_sha256","0"*64),"check_false":lambda e,l,o,a:e["authorization_result"]["checks"].__setitem__("r7p_pass18",False),"observed_missing":lambda e,l,o,a:e["observed"].pop("r7p_result_sha256"),"lock_hash":lambda e,l,o,a:e.__setitem__("lock_sha256","0"*64),"lock_closed":lambda e,l,o,a:l.__setitem__("execution_open",False),"r7p":lambda e,l,o,a:a.__setitem__("r7p_result_sha256","0"*64),"stages":lambda e,l,o,a:e.__setitem__("outer_failure_stages",list(reversed(verifier.OUTER_STAGES)))}
            for name, mutate in mutations.items():
                e,l,o,a = map(copy.deepcopy,(extension,lock,observed,auth)); mutate(e,l,o,a)
                if not verifier.extension_valid(e,l,o,a): rejected.append(name)
            return baseline and set(rejected) == set(mutations), rejected
        finally: verifier.LOCK = old_lock


def clean_state() -> tuple[bool, dict]:
    required = {"r7a_auth_result": runner.AUTH_RESULT.exists() and sha256(runner.AUTH_RESULT) == runner.AUTH_RESULT_SHA, "r7p_result": R7P_RESULT.exists() and sha256(R7P_RESULT) == runner.R7P_SHA}
    absent_paths = {"r7a_out": runner.physical.OUT, "r7a_failed": runner.physical.FAILED, "r7a_quarantine": runner.physical.QUAR, "r7c2_out": runner.REVISION_OUT, "r7c2_failed": runner.OUTER_FAILED, "r7c2_quarantine": runner.OUTER_QUARANTINE, "r7c2_preflight_result": RESULT, "r7c2_verification": verifier.VERIFY}
    absent = {name: not path.exists() for name, path in absent_paths.items()}
    globs = {"r7a_inprogress": list(R.glob(runner.physical.OUT.name + ".*.inprogress")), "r7a_failure_inprogress": list(R.glob(runner.physical.FAILED.name + ".*.inprogress")), "r7c2_inprogress": list(R.glob(runner.REVISION_OUT.name + ".*.inprogress")), "r7c2_failure_inprogress": list(R.glob(runner.OUTER_FAILED.name + ".*.inprogress")), "all_r7a_r7c2_inprogress": [path for path in R.glob("*.inprogress") if "r7a" in path.name or "r7c2" in path.name]}
    glob_empty = {name: len(paths) == 0 for name, paths in globs.items()}
    evidence = {"required": required, "absent": absent, "glob_empty": glob_empty, "glob_paths": {name: [str(path) for path in paths] for name, paths in globs.items()}}
    return all(required.values()) and all(absent.values()) and all(glob_empty.values()), evidence


def candidate_import_free() -> bool:
    tree = ast.parse(Path(verifier.__file__).read_text()); forbidden = {"run_het_next_l0_ph1_intel_execution_r7c2", "run_het_next_l0_ph1_intel_execution_r7c1", "run_het_next_l0_ph1_intel_execution_r7c", "run_het_next_l0_ph1_intel_execution_r7b"}; names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import): names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom): names.add(node.module or "")
    return not (names & forbidden)


def no_device_static() -> bool:
    tree = ast.parse(Path(__file__).read_text()); forbidden_modules = {"pyopencl","cupy","torch","safetensors","transformers"}; forbidden_calls = {"WinDLL","CDLL","LoadLibrary","Backend","execute_authorized"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(alias.name.split(".")[0] in forbidden_modules for alias in node.names): return False
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] in forbidden_modules: return False
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ""
            if name in forbidden_calls: return False
    return True


def main() -> int:
    lock = json.loads(LOCK.read_text()); observed = {name: sha256(path) for name, path in runner.CHAIN.items()}; old = install_prior_fixture_modules()
    try: inherited_cases = {name: prior_preflight.case(name) for name in ("early_raise","structured3","bare3","multiple3","oversized3","stale","positive","negative","success_without_commit")}
    finally: restore_prior_fixture_modules(old)
    true_rc, true_summary = device_opened_case(True); false_rc, false_summary = device_opened_case(False); wrong_rc, wrong_summary = device_opened_case("true")
    device_checks = {"true_retained": true_rc and true_summary.get("device_opened") is True and true_summary.get("inherited",[{}])[0].get("inherited_device_opened") is True and true_summary.get("inherited_evidence_valid") is True, "false_retained": false_rc and false_summary.get("device_opened") is False and false_summary.get("inherited",[{}])[0].get("inherited_device_opened") is False and false_summary.get("inherited_evidence_valid") is True, "wrong_type_rejected": wrong_rc and wrong_summary.get("device_opened") is False and wrong_summary.get("inherited",[{}])[0].get("inherited_device_opened") is None and wrong_summary.get("inherited_evidence_valid") is False}
    extension_ok, rejected = extension_mutations(); clean_ok, clean_evidence = clean_state()
    checks = {"hash_bindings": all(lock.get(name) == digest for name,digest in observed.items()), "closed_pending": lock.get("kind") == "ph1_intel_execution_r7c2_lock" and lock.get("execution_open") is False and lock.get("audit_token") == "PENDING", "auth_result_exact": runner.validate_auth_result()["pass"] is True, "inherited_r7c1_cases": all(inherited_cases.values()), "device_opened_exact": all(device_checks.values()), "extension_mutations": extension_ok and len(rejected) == 8, "candidate_import_free": candidate_import_free(), "no_device_static": no_device_static(), "clean_state_complete": clean_ok}
    output = {"kind": "ph1_intel_execution_r7c2_static_preflight", "checks": checks, "pass": all(checks.values()), "passed": sum(checks.values()), "total": len(checks), "no_payload_compiler_device": True, "inherited_cases": inherited_cases, "device_opened_cases": device_checks, "clean_state": clean_evidence, "rejected_extension_mutations": rejected}
    if RESULT.exists(): raise FileExistsError(RESULT)
    RESULT.write_text(json.dumps(output,sort_keys=True,indent=2)+"\n"); print(json.dumps(output,indent=2)); return 0 if output["pass"] else 3


if __name__ == "__main__": raise SystemExit(main())
