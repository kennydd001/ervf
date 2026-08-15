#!/usr/bin/env python3
"""PH1 Intel R7C: closed authorization gate with an outer failure boundary."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
S = ROOT / "scripts/streamq5_moe"
R = ROOT / "reports/streamq5_moe"
sys.path.insert(0, str(S))

import run_het_next_l0_ph1_intel_execution_r7a as physical

LOCK = R / "het_next_l0_ph1_intel_execution_r7c_lock.json"
PREREG = R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R7C_PREREGISTRATION_2026-08-14.md"
PREFLIGHT = S / "preflight_het_next_l0_ph1_intel_execution_r7c.py"
VERIFIER = S / "verify_het_next_l0_ph1_intel_execution_r7c.py"
AUTH_RESULT = R / "het_next_l0_ph1_intel_execution_r7a_authorization_preflight.json"
OUTER_FAILED = R / "het_next_l0_ph1_intel_execution_r7c_failed_attempts"
OUTER_QUARANTINE = R / "het_next_l0_ph1_intel_execution_r7c_quarantine"
ACK = "PH1_INTEL_EXECUTION_R7C_AFTER_FAILURE_AND_INDEPENDENCE_AUDIT_GO"
R7A_ACK = "PH1_INTEL_EXECUTION_R7A_AFTER_R7P_PASS18_AND_FINAL_AUDIT_GO"
R7P_SHA = "e10c513fdbecb27e08319c462ba1d1020b1c94c4ff5d9199047ae513197dd959"
AUTH_RESULT_SHA = "a5b8e70cd40e241e16a250347cf06258a6540100f40423bc7216cb3639191265"
MAX_FAILURE_BYTES = 16 * 2**20
AUTH_CHECKS = {
    "authorization_only_runner",
    "fixed_verifier_unchanged",
    "hash_bindings",
    "no_device_static",
    "open_exact",
    "output_absent",
    "r7p_pass18",
}
OUTER_STAGES = (
    "psutil_import",
    "start_ram",
    "payload",
    "post_payload_resource",
    "predevice",
    "device_execute",
    "serialize_commit",
)

CHAIN = {
    "runner_sha256": Path(__file__),
    "verifier_sha256": VERIFIER,
    "preflight_sha256": PREFLIGHT,
    "prereg_sha256": PREREG,
    "r7b_runner_sha256": S / "run_het_next_l0_ph1_intel_execution_r7b.py",
    "r7b_verifier_sha256": S / "verify_het_next_l0_ph1_intel_execution_r7b.py",
    "r7b_prereg_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R7B_PREREGISTRATION_2026-08-14.md",
    "r7b_lock_sha256": R / "het_next_l0_ph1_intel_execution_r7b_lock.json",
    "r7b_audit_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R7B_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md",
    "physical_runner_sha256": S / "run_het_next_l0_ph1_intel_execution_r7a.py",
    "physical_verifier_sha256": S / "verify_het_next_l0_ph1_intel_execution_r7a.py",
    "physical_preflight_sha256": S / "preflight_het_next_l0_ph1_intel_execution_r7a.py",
    "physical_prereg_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R7A_PREREGISTRATION_2026-08-14.md",
    "physical_lock_sha256": R / "het_next_l0_ph1_intel_execution_r7a_lock.json",
    "authorization_result_sha256": AUTH_RESULT,
    "r7a_audit_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R7A_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md",
    "r7p_preflight_sha256": S / "preflight_het_next_l0_ph1_intel_execution_r7p.py",
    "r7p_lock_sha256": R / "het_next_l0_ph1_intel_execution_r7p_lock.json",
    "r7p_prereg_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R7P_PREREGISTRATION_2026-08-14.md",
    "r7p_audit_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R7P_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md",
    "r7p_result_sha256": R / "het_next_l0_ph1_intel_execution_r7p_static_preflight.json",
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
    checks = result.get("checks")
    if not (
        set(result) == {"ack", "checks", "kind", "no_payload_compiler_device", "pass", "passed", "r7p_result_sha256", "total"}
        and result.get("kind") == "ph1_intel_execution_r7a_authorization_preflight"
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
    if not (
        set(lock) == {"kind", "execution_open", "audit_token", "physical_output", "physical_verifier", *observed.keys()}
        and lock.get("kind") == "ph1_intel_execution_r7c_lock"
        and lock.get("execution_open") is True
        and lock.get("audit_token") == ACK
        and lock.get("physical_output") == "het_next_l0_ph1_intel_execution_r7a"
        and lock.get("physical_verifier") == "verify_het_next_l0_ph1_intel_execution_r7c.py"
        and all(lock.get(name) == digest for name, digest in observed.items())
        and observed["authorization_result_sha256"] == AUTH_RESULT_SHA
        and observed["r7b_audit_sha256"] == "20ccffff336cb57f855749e4cd1ee9e0901b39c8df91d07447ebea3d9fe902cd"
        and observed["r7p_result_sha256"] == R7P_SHA
    ):
        raise RuntimeError("r7c_authorization")
    inherited = physical.authorize()
    inherited["r7c_authorization"] = {
        "lock_sha256": sha256(LOCK),
        "observed": observed,
        "authorization_result_sha256": AUTH_RESULT_SHA,
        "authorization_result": result,
        "audit_token": ACK,
        "outer_failure_stages": list(OUTER_STAGES),
    }
    return inherited


def _canonical(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def _bounded_payload(payload: dict) -> bytes:
    encoded = _canonical(payload)
    if len(encoded) <= MAX_FAILURE_BYTES:
        return encoded
    summary = {
        "kind": "ph1_intel_execution_r7c_failure",
        "status": "valid_negative_failure",
        "stage": payload.get("stage", "outer_boundary"),
        "error": "failure_evidence_oversize",
        "original_bytes": len(encoded),
        "original_sha256": hashlib.sha256(encoded).hexdigest(),
        "device_opened": payload.get("device_opened", False),
        "disposition": "bounded_summary_only",
    }
    bounded = _canonical(summary)
    if len(bounded) > MAX_FAILURE_BYTES:
        raise RuntimeError("bounded_failure_impossible")
    return bounded


def atomic_outer_failure(payload: dict) -> Path:
    encoded = _bounded_payload(payload)
    OUTER_FAILED.mkdir(parents=True, exist_ok=True)
    nonce = uuid.uuid4().hex
    temporary = R / f"{OUTER_FAILED.name}.{nonce}.inprogress"
    destination = OUTER_FAILED / f"attempt_{nonce}"
    temporary.mkdir(parents=False, exist_ok=False)
    try:
        physical.base.write(temporary / "failure.json", encoded)
        physical.base.move(temporary, destination)
    except Exception:
        if temporary.exists():
            OUTER_QUARANTINE.mkdir(parents=True, exist_ok=True)
            physical.base.move(temporary, OUTER_QUARANTINE / f"failed_commit_{nonce}")
        raise
    return destination


def quarantine_stale_outer_failures() -> None:
    stale = sorted(R.glob(OUTER_FAILED.name + ".*.inprogress"))
    if not stale:
        return
    OUTER_QUARANTINE.mkdir(parents=True, exist_ok=True)
    for path in stale:
        physical.base.move(path, OUTER_QUARANTINE / ("stale_" + path.name.rsplit(".", 2)[-2]))
    raise RuntimeError("stale_outer_failure_quarantined")


def outer_execute(authorization: dict, executor=physical.execute_authorized) -> int:
    try:
        quarantine_stale_outer_failures()
        return executor(authorization)
    except Exception as exc:
        # A valid immutable physical commit is never polluted by wrapper failure evidence.
        if physical.OUT.exists():
            try:
                existing = physical.verify_bundle(physical.OUT)
                return 0 if existing.get("positive") is True else 3
            except Exception:
                pass
        stage = getattr(exc, "stage", "r7a_outer_boundary")
        payload = {
            "kind": "ph1_intel_execution_r7c_failure",
            "status": "valid_negative_failure",
            "stage": stage,
            "error": f"{type(exc).__name__}:{exc}"[:2048],
            "traceback": traceback.format_exc()[-32768:],
            "device_opened": bool(getattr(exc, "device_opened", False)),
            "covered_stages": list(OUTER_STAGES),
            "disposition": "atomic_create_new_bounded_outer_failure",
        }
        atomic_outer_failure(payload)
        return 3


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
    return outer_execute(authorization)


if __name__ == "__main__":
    raise SystemExit(main())
