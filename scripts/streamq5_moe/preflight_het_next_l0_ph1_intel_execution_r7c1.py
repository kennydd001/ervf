#!/usr/bin/env python3
"""Closed no-device R7C1 preflight for delegated-return lifecycle semantics."""
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
import run_het_next_l0_ph1_intel_execution_r7c1 as runner
import verify_het_next_l0_ph1_intel_execution_r7c1 as verifier

LOCK = R / "het_next_l0_ph1_intel_execution_r7c1_lock.json"
RESULT = R / "het_next_l0_ph1_intel_execution_r7c1_static_preflight.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configure(root: Path) -> tuple:
    old = (runner.R, runner.OUTER_FAILED, runner.OUTER_QUARANTINE, runner.physical.OUT, runner.physical.FAILED)
    runner.R = root; runner.OUTER_FAILED = root / "r7c1_failed"; runner.OUTER_QUARANTINE = root / "r7c1_quarantine"
    runner.physical.OUT = root / "r7a_output"; runner.physical.FAILED = root / "r7a_failed"
    return old


def restore(old: tuple) -> None:
    runner.R, runner.OUTER_FAILED, runner.OUTER_QUARANTINE, runner.physical.OUT, runner.physical.FAILED = old


def inherited_failure(disposition="atomic_create_new_failure_only", oversized=False) -> Path:
    directory = runner.physical.FAILED / ("attempt_" + str(len(list(runner.physical.FAILED.glob("*")))))
    directory.mkdir(parents=True)
    payload = {"kind": "ph1_intel_execution_r7a_failure", "status": "valid_negative_failure", "error": "injected", "device_opened": False, "disposition": disposition}
    data = runner.canonical(payload)
    if oversized:
        data += b" " * (runner.MAX_FAILURE_BYTES + 1)
    (directory / "failure.json").write_bytes(data)
    return directory / "failure.json"


def committed(positive: bool) -> None:
    runner.physical.OUT.mkdir(parents=True)
    result = runner.physical.base.canon({"kind": "ph1_intel_execution_r7a", "positive": positive})
    runner.physical.base.write(runner.physical.OUT / "result.json", result)
    files = [{"name": "result.json", "bytes": len(result), "sha256": hashlib.sha256(result).hexdigest()}]
    runner.physical.base.write(runner.physical.OUT / "manifest.json", runner.physical.base.canon({"kind": "ph1_intel_execution_r7a_manifest", "files": files}))
    runner.physical.base.write(runner.physical.OUT / "commit.json", runner.physical.base.canon({"kind": "ph1_intel_execution_r7a_commit", "manifest_sha256": sha256(runner.physical.OUT / "manifest.json"), "result_sha256": sha256(runner.physical.OUT / "result.json")}))


def one_summary() -> dict:
    paths = list(runner.OUTER_FAILED.glob("*/failure.json"))
    if len(paths) != 1:
        raise RuntimeError("summary_count")
    if paths[0].stat().st_size > runner.MAX_FAILURE_BYTES:
        raise RuntimeError("summary_cap")
    return json.loads(paths[0].read_text())


def case(mode: str) -> bool:
    with tempfile.TemporaryDirectory() as temporary:
        old = configure(Path(temporary))
        try:
            if mode == "early_raise":
                class Early(RuntimeError): stage = "payload"; device_opened = False
                def executor(_): raise Early("injected")
                return runner.outer_execute({}, executor) == 3 and one_summary().get("stage") == "payload"
            if mode == "structured3":
                def executor(_): inherited_failure(); return 3
                rc = runner.outer_execute({}, executor); row = one_summary()
                return rc == 3 and row.get("inherited_evidence_valid") is True and row.get("new_inherited_failure_count") == 1 and row.get("adjudication") == "one_valid_inherited_failure"
            if mode == "bare3":
                rc = runner.outer_execute({}, lambda _: 3); row = one_summary()
                return rc == 3 and row.get("inherited_evidence_valid") is False and row.get("adjudication") == "missing_inherited_failure"
            if mode == "multiple3":
                def executor(_): inherited_failure(); inherited_failure(); return 3
                rc = runner.outer_execute({}, executor); row = one_summary()
                return rc == 3 and row.get("new_inherited_failure_count") == 2 and row.get("adjudication") == "multiple_inherited_failures"
            if mode == "oversized3":
                def executor(_): inherited_failure(oversized=True); return 3
                rc = runner.outer_execute({}, executor); row = one_summary()
                return rc == 3 and row.get("inherited_evidence_valid") is False and row["inherited"][0].get("adjudication") == "oversize_or_cardinality"
            if mode == "stale":
                (Path(temporary) / (runner.OUTER_FAILED.name + ".fixture.inprogress")).mkdir()
                rc = runner.outer_execute({}, lambda _: 0); row = one_summary()
                return rc == 3 and row.get("stage") == "r7a_outer_boundary" and len(list(runner.OUTER_QUARANTINE.iterdir())) == 1
            if mode == "positive":
                committed(True); return runner.outer_execute({}, lambda _: 0) == 0 and not runner.OUTER_FAILED.exists()
            if mode == "negative":
                committed(False); return runner.outer_execute({}, lambda _: 3) == 3 and not runner.OUTER_FAILED.exists()
            if mode == "success_without_commit":
                rc = runner.outer_execute({}, lambda _: 0); row = one_summary()
                return rc == 3 and row.get("stage") == "delegated_invalid_success"
            return False
        finally:
            restore(old)


def extension_mutations() -> tuple[bool, list[str]]:
    observed = {name: sha256(path) for name, path in verifier.CHAIN.items()}; auth = json.loads(verifier.AUTH_RESULT.read_text())
    with tempfile.TemporaryDirectory() as temporary:
        old_lock = verifier.LOCK
        try:
            verifier.LOCK = Path(temporary) / "lock.json"; lock = json.loads(LOCK.read_text()); lock["execution_open"] = True; lock["audit_token"] = verifier.ACK
            verifier.LOCK.write_text(json.dumps(lock, sort_keys=True, indent=2) + "\n")
            extension = {"lock_sha256": sha256(verifier.LOCK), "observed": observed, "authorization_result_sha256": verifier.AUTH_RESULT_SHA, "authorization_result": auth, "audit_token": verifier.ACK, "outer_failure_stages": verifier.OUTER_STAGES}
            baseline = verifier.extension_valid(extension, lock, observed, auth); rejected = []
            mutations = {
                "token": lambda e,l,o,a:e.__setitem__("audit_token","bad"),
                "result_hash": lambda e,l,o,a:e.__setitem__("authorization_result_sha256","0"*64),
                "check_false": lambda e,l,o,a:e["authorization_result"]["checks"].__setitem__("r7p_pass18",False),
                "observed_missing": lambda e,l,o,a:e["observed"].pop("r7p_result_sha256"),
                "lock_hash": lambda e,l,o,a:e.__setitem__("lock_sha256","0"*64),
                "lock_closed": lambda e,l,o,a:l.__setitem__("execution_open",False),
                "r7p": lambda e,l,o,a:a.__setitem__("r7p_result_sha256","0"*64),
                "stages": lambda e,l,o,a:e.__setitem__("outer_failure_stages",list(reversed(verifier.OUTER_STAGES))),
            }
            for name, mutate in mutations.items():
                e,l,o,a = map(copy.deepcopy, (extension, lock, observed, auth)); mutate(e,l,o,a)
                if not verifier.extension_valid(e,l,o,a): rejected.append(name)
            return baseline and set(rejected) == set(mutations), rejected
        finally:
            verifier.LOCK = old_lock


def candidate_import_free() -> bool:
    tree = ast.parse(Path(verifier.__file__).read_text()); forbidden = {"run_het_next_l0_ph1_intel_execution_r7c1", "run_het_next_l0_ph1_intel_execution_r7c", "run_het_next_l0_ph1_intel_execution_r7b"}
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import): names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom): names.add(node.module or "")
    return not (names & forbidden)


def no_device_static() -> bool:
    tree = ast.parse(Path(__file__).read_text())
    forbidden_modules = {"pyopencl", "cupy", "torch", "safetensors", "transformers"}
    forbidden_calls = {"WinDLL", "CDLL", "LoadLibrary", "Backend", "execute_authorized"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(alias.name.split(".")[0] in forbidden_modules for alias in node.names):
            return False
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] in forbidden_modules:
            return False
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ""
            if name in forbidden_calls:
                return False
    return True


def main() -> int:
    lock = json.loads(LOCK.read_text()); observed = {name: sha256(path) for name, path in runner.CHAIN.items()}; extension_ok, rejected = extension_mutations()
    cases = {name: case(name) for name in ("early_raise", "structured3", "bare3", "multiple3", "oversized3", "stale", "positive", "negative", "success_without_commit")}
    checks = {
        "hash_bindings": all(lock.get(name) == digest for name, digest in observed.items()),
        "closed_pending": lock.get("kind") == "ph1_intel_execution_r7c1_lock" and lock.get("execution_open") is False and lock.get("audit_token") == "PENDING",
        "auth_result_exact": sha256(runner.AUTH_RESULT) == runner.AUTH_RESULT_SHA and runner.validate_auth_result()["pass"] is True,
        "delegated_return_cases": all(cases.values()),
        "extension_baseline_mutations": extension_ok and len(rejected) == 8,
        "candidate_import_free": candidate_import_free(),
        "no_device_static": no_device_static(),
        "outputs_absent": not runner.physical.OUT.exists() and not RESULT.exists() and not verifier.VERIFY.exists(),
    }
    output = {"kind": "ph1_intel_execution_r7c1_static_preflight", "checks": checks, "pass": all(checks.values()), "passed": sum(checks.values()), "total": len(checks), "no_payload_compiler_device": True, "delegated_cases": cases, "rejected_extension_mutations": rejected}
    if RESULT.exists(): raise FileExistsError(RESULT)
    RESULT.write_text(json.dumps(output, sort_keys=True, indent=2) + "\n"); print(json.dumps(output, indent=2)); return 0 if output["pass"] else 3


if __name__ == "__main__": raise SystemExit(main())
