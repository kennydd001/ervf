#!/usr/bin/env python3
"""Closed no-device R7C preflight: authorization and outer-failure simulations."""
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

import run_het_next_l0_ph1_intel_execution_r7c as runner
import verify_het_next_l0_ph1_intel_execution_r7c as verifier

LOCK = R / "het_next_l0_ph1_intel_execution_r7c_lock.json"
RESULT = R / "het_next_l0_ph1_intel_execution_r7c_static_preflight.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Injected(RuntimeError):
    def __init__(self, stage: str):
        super().__init__("injected_" + stage)
        self.stage = stage
        self.device_opened = stage in {"device_execute", "serialize_commit"}


def outer_failure_simulation() -> bool:
    old = (runner.R, runner.OUTER_FAILED, runner.OUTER_QUARANTINE, runner.physical.OUT)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        try:
            runner.R = root
            runner.OUTER_FAILED = root / "failed"
            runner.OUTER_QUARANTINE = root / "quarantine"
            runner.physical.OUT = root / "physical_output"
            for stage in runner.OUTER_STAGES:
                def fail(_authorization, selected=stage):
                    raise Injected(selected)
                if runner.outer_execute({"fixture": True}, fail) != 3:
                    return False
            failures = sorted(runner.OUTER_FAILED.glob("*/failure.json"))
            if len(failures) != len(runner.OUTER_STAGES):
                return False
            rows = [json.loads(path.read_text()) for path in failures]
            if {row.get("stage") for row in rows} != set(runner.OUTER_STAGES):
                return False
            if not all(row.get("kind") == "ph1_intel_execution_r7c_failure" and row.get("status") == "valid_negative_failure" and row.get("disposition") == "atomic_create_new_bounded_outer_failure" for row in rows):
                return False
            bounded = runner._bounded_payload({"kind": "ph1_intel_execution_r7c_failure", "stage": "serialize_commit", "detail": "x" * (runner.MAX_FAILURE_BYTES + 1)})
            if len(bounded) > runner.MAX_FAILURE_BYTES or json.loads(bounded).get("disposition") != "bounded_summary_only":
                return False
            stale = root / (runner.OUTER_FAILED.name + ".fixture.inprogress")
            stale.mkdir()
            try:
                runner.quarantine_stale_outer_failures()
                return False
            except RuntimeError as exc:
                if str(exc) != "stale_outer_failure_quarantined":
                    return False
            if stale.exists() or len(list(runner.OUTER_QUARANTINE.iterdir())) != 1:
                return False
            # A valid committed R7A result wins over an outer exception and is not polluted.
            runner.physical.OUT.mkdir()
            result_bytes = runner.physical.base.canon({"kind": "ph1_intel_execution_r7a", "positive": True})
            runner.physical.base.write(runner.physical.OUT / "result.json", result_bytes)
            files = [{"name": "result.json", "bytes": len(result_bytes), "sha256": hashlib.sha256(result_bytes).hexdigest()}]
            runner.physical.base.write(runner.physical.OUT / "manifest.json", runner.physical.base.canon({"kind": "ph1_intel_execution_r7a_manifest", "files": files}))
            runner.physical.base.write(runner.physical.OUT / "commit.json", runner.physical.base.canon({
                "kind": "ph1_intel_execution_r7a_commit",
                "manifest_sha256": sha256(runner.physical.OUT / "manifest.json"),
                "result_sha256": sha256(runner.physical.OUT / "result.json"),
            }))
            before = len(list(runner.OUTER_FAILED.glob("*/failure.json")))
            def fail_after_commit(_authorization):
                raise Injected("serialize_commit")
            if runner.outer_execute({"fixture": True}, fail_after_commit) != 0:
                return False
            if len(list(runner.OUTER_FAILED.glob("*/failure.json"))) != before:
                return False
            return True
        finally:
            runner.R, runner.OUTER_FAILED, runner.OUTER_QUARANTINE, runner.physical.OUT = old


def extension_mutations() -> tuple[bool, list[str]]:
    observed = {name: sha256(path) for name, path in verifier.CHAIN.items()}
    auth = json.loads(verifier.AUTH_RESULT.read_text())
    with tempfile.TemporaryDirectory() as temporary:
        old_lock = verifier.LOCK
        try:
            verifier.LOCK = Path(temporary) / "open_lock.json"
            open_lock = json.loads(LOCK.read_text())
            open_lock["execution_open"] = True
            open_lock["audit_token"] = verifier.ACK
            verifier.LOCK.write_text(json.dumps(open_lock, sort_keys=True, indent=2) + "\n")
            extension = {
                "lock_sha256": sha256(verifier.LOCK),
                "observed": observed,
                "authorization_result_sha256": verifier.AUTH_RESULT_SHA,
                "authorization_result": auth,
                "audit_token": verifier.ACK,
                "outer_failure_stages": verifier.OUTER_STAGES,
            }
            baseline = verifier.extension_valid(extension, open_lock, observed, auth)
            rejected = []
            mutations = {
                "wrong_token": (lambda e, l, o, a: e.__setitem__("audit_token", "WRONG")),
                "wrong_result_hash": (lambda e, l, o, a: e.__setitem__("authorization_result_sha256", "0" * 64)),
                "false_result_check": (lambda e, l, o, a: e["authorization_result"]["checks"].__setitem__("r7p_pass18", False)),
                "missing_observed": (lambda e, l, o, a: e["observed"].pop("r7p_result_sha256")),
                "wrong_lock_hash": (lambda e, l, o, a: e.__setitem__("lock_sha256", "0" * 64)),
                "closed_lock": (lambda e, l, o, a: l.__setitem__("execution_open", False)),
                "wrong_r7p": (lambda e, l, o, a: a.__setitem__("r7p_result_sha256", "0" * 64)),
                "wrong_stage_order": (lambda e, l, o, a: e.__setitem__("outer_failure_stages", list(reversed(verifier.OUTER_STAGES)))),
            }
            for name, mutate in mutations.items():
                candidate_extension = copy.deepcopy(extension)
                candidate_lock = copy.deepcopy(open_lock)
                candidate_observed = copy.deepcopy(observed)
                candidate_auth = copy.deepcopy(auth)
                mutate(candidate_extension, candidate_lock, candidate_observed, candidate_auth)
                if not verifier.extension_valid(candidate_extension, candidate_lock, candidate_observed, candidate_auth):
                    rejected.append(name)
            return baseline and set(rejected) == set(mutations), rejected
        finally:
            verifier.LOCK = old_lock


def independent_verifier_ast() -> bool:
    tree = ast.parse(Path(verifier.__file__).read_text())
    forbidden = {
        "run_het_next_l0_ph1_intel_execution_r7b",
        "run_het_next_l0_ph1_intel_execution_r7c",
    }
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(item.name for item in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    return not any(name in forbidden for name in imported)


def no_device_ast() -> bool:
    paths = (Path(__file__), Path(runner.__file__), Path(verifier.__file__))
    forbidden_modules = {"pyopencl", "cupy", "torch", "safetensors", "transformers"}
    forbidden_calls = {"WinDLL", "CDLL", "LoadLibrary", "Backend", "execute_authorized"}
    for path in paths:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [item.name.split(".")[0] for item in node.names] if isinstance(node, ast.Import) else [(node.module or "").split(".")[0]]
                if forbidden_modules.intersection(names):
                    return False
            if path == Path(__file__) and isinstance(node, ast.Call):
                name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ""
                if name in forbidden_calls:
                    return False
    return True


def main() -> int:
    lock = json.loads(LOCK.read_text())
    observed = {name: sha256(path) for name, path in runner.CHAIN.items()}
    extension_ok, rejected = extension_mutations()
    checks = {
        "hash_bindings": all(lock.get(name) == digest for name, digest in observed.items()),
        "closed_pending": lock.get("kind") == "ph1_intel_execution_r7c_lock" and lock.get("execution_open") is False and lock.get("audit_token") == "PENDING",
        "auth_result_exact": sha256(runner.AUTH_RESULT) == runner.AUTH_RESULT_SHA and runner.validate_auth_result()["pass"] is True,
        "outer_failure_simulation": outer_failure_simulation(),
        "extension_baseline_and_mutations": extension_ok and len(rejected) == 8,
        "independent_verifier_ast": independent_verifier_ast(),
        "no_device_static": no_device_ast(),
        "outputs_absent": not runner.physical.OUT.exists() and not RESULT.exists() and not verifier.VERIFY.exists(),
    }
    output = {
        "kind": "ph1_intel_execution_r7c_static_preflight",
        "checks": checks,
        "pass": all(checks.values()),
        "passed": sum(value is True for value in checks.values()),
        "total": len(checks),
        "no_payload_compiler_device": True,
        "rejected_extension_mutations": rejected,
    }
    if RESULT.exists():
        raise FileExistsError(RESULT)
    RESULT.write_text(json.dumps(output, sort_keys=True, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    return 0 if output["pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
