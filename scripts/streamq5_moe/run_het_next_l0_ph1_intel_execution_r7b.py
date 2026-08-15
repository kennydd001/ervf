#!/usr/bin/env python3
"""PH1 Intel R7B: authorization-result gate around immutable R7A execution."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
S = ROOT / "scripts/streamq5_moe"
R = ROOT / "reports/streamq5_moe"
sys.path.insert(0, str(S))

import run_het_next_l0_ph1_intel_execution_r7a as physical

LOCK = R / "het_next_l0_ph1_intel_execution_r7b_lock.json"
PREREG = R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R7B_PREREGISTRATION_2026-08-14.md"
VERIFIER = S / "verify_het_next_l0_ph1_intel_execution_r7b.py"
AUTH_RESULT = R / "het_next_l0_ph1_intel_execution_r7a_authorization_preflight.json"
R7A_AUDIT = R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R7A_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md"
R7P_RESULT = R / "het_next_l0_ph1_intel_execution_r7p_static_preflight.json"
ACK = "PH1_INTEL_EXECUTION_R7B_AFTER_R7A_PASS7_AND_AUTH_AUDIT_GO"
R7A_ACK = "PH1_INTEL_EXECUTION_R7A_AFTER_R7P_PASS18_AND_FINAL_AUDIT_GO"
R7P_SHA = "e10c513fdbecb27e08319c462ba1d1020b1c94c4ff5d9199047ae513197dd959"
AUTH_RESULT_SHA = "a5b8e70cd40e241e16a250347cf06258a6540100f40423bc7216cb3639191265"
AUTH_CHECKS = {
    "authorization_only_runner",
    "fixed_verifier_unchanged",
    "hash_bindings",
    "no_device_static",
    "open_exact",
    "output_absent",
    "r7p_pass18",
}

CHAIN = {
    "runner_sha256": Path(__file__),
    "verifier_sha256": VERIFIER,
    "prereg_sha256": PREREG,
    "physical_runner_sha256": S / "run_het_next_l0_ph1_intel_execution_r7a.py",
    "physical_verifier_sha256": S / "verify_het_next_l0_ph1_intel_execution_r7a.py",
    "physical_preflight_sha256": S / "preflight_het_next_l0_ph1_intel_execution_r7a.py",
    "physical_prereg_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R7A_PREREGISTRATION_2026-08-14.md",
    "physical_lock_sha256": R / "het_next_l0_ph1_intel_execution_r7a_lock.json",
    "authorization_result_sha256": AUTH_RESULT,
    "r7a_audit_sha256": R7A_AUDIT,
    "r7p_preflight_sha256": S / "preflight_het_next_l0_ph1_intel_execution_r7p.py",
    "r7p_lock_sha256": R / "het_next_l0_ph1_intel_execution_r7p_lock.json",
    "r7p_prereg_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R7P_PREREGISTRATION_2026-08-14.md",
    "r7p_audit_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R7P_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md",
    "r7p_result_sha256": R7P_RESULT,
    "r7_runner_sha256": S / "run_het_next_l0_ph1_intel_execution_r7.py",
    "r7_verifier_sha256": S / "verify_het_next_l0_ph1_intel_execution_r7.py",
    "backend_sha256": S / "het_next_l0_ph1_intel_execution_r6_backend.py",
    "common_sha256": S / "het_next_l0_ph1_intel_execution_r6_common.py",
    "r0_backend_sha256": S / "het_next_l0_ph1_intel_execution_r0_backend.py",
    "r0_runner_sha256": S / "run_het_next_l0_ph1_intel_execution_r0.py",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_auth_result() -> dict:
    if sha256(AUTH_RESULT) != AUTH_RESULT_SHA:
        raise RuntimeError("authorization_result_hash")
    result = json.loads(AUTH_RESULT.read_text())
    expected_keys = {
        "ack", "checks", "kind", "no_payload_compiler_device", "pass",
        "passed", "r7p_result_sha256", "total",
    }
    if set(result) != expected_keys:
        raise RuntimeError("authorization_result_schema")
    checks = result.get("checks")
    if not (
        result.get("kind") == "ph1_intel_execution_r7a_authorization_preflight"
        and result.get("pass") is True
        and result.get("passed") == result.get("total") == 7
        and result.get("no_payload_compiler_device") is True
        and result.get("ack") == R7A_ACK
        and result.get("r7p_result_sha256") == R7P_SHA
        and isinstance(checks, dict)
        and set(checks) == AUTH_CHECKS
        and all(value is True for value in checks.values())
    ):
        raise RuntimeError("authorization_result_contract")
    return result


def authorize() -> dict:
    result = validate_auth_result()
    observed = {name: sha256(path) for name, path in CHAIN.items()}
    lock = json.loads(LOCK.read_text())
    expected_lock_keys = {
        "kind", "execution_open", "audit_token", "physical_output",
        "physical_verifier", *observed.keys(),
    }
    if not (
        set(lock) == expected_lock_keys
        and lock.get("kind") == "ph1_intel_execution_r7b_lock"
        and lock.get("execution_open") is True
        and lock.get("audit_token") == ACK
        and lock.get("physical_output") == "het_next_l0_ph1_intel_execution_r7a"
        and lock.get("physical_verifier") == "verify_het_next_l0_ph1_intel_execution_r7b.py"
        and all(lock.get(name) == digest for name, digest in observed.items())
        and observed["authorization_result_sha256"] == AUTH_RESULT_SHA
        and observed["r7a_audit_sha256"] == "cbcbd1a861fc54e0dd529de22eb8fd3658a7fa81292e2c0ae0b188366055a5cd"
        and observed["r7p_result_sha256"] == R7P_SHA
    ):
        raise RuntimeError("r7b_authorization")
    # This replays R7A's complete transitive compile/CPU/R7P authorization.
    inherited = physical.authorize()
    inherited["r7b_authorization"] = {
        "lock_sha256": sha256(LOCK),
        "observed": observed,
        "authorization_result_sha256": AUTH_RESULT_SHA,
        "authorization_result": result,
        "audit_token": ACK,
    }
    return inherited


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ack", required=True)
    args = parser.parse_args()
    if args.ack != ACK:
        return 3
    try:
        authorization = authorize()
    except Exception:
        return 3
    # No configure/recover/payload/OpenCL path is reachable before authorize passes.
    return physical.execute_authorized(authorization)


if __name__ == "__main__":
    raise SystemExit(main())
