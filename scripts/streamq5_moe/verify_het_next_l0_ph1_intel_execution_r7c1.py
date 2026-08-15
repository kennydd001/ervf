#!/usr/bin/env python3
"""Independent R7C1 authorization verifier; frozen R7A numerical replay follows."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
S = ROOT / "scripts/streamq5_moe"
R = ROOT / "reports/streamq5_moe"
OUT = R / "het_next_l0_ph1_intel_execution_r7a"
VERIFY = R / "het_next_l0_ph1_intel_execution_r7c1_independent_verification.json"
LOCK = R / "het_next_l0_ph1_intel_execution_r7c1_lock.json"
AUTH_RESULT = R / "het_next_l0_ph1_intel_execution_r7a_authorization_preflight.json"
ACK = "PH1_INTEL_EXECUTION_R7C1_AFTER_RETURN_ADJUDICATION_AUDIT_GO"
R7A_ACK = "PH1_INTEL_EXECUTION_R7A_AFTER_R7P_PASS18_AND_FINAL_AUDIT_GO"
R7P_SHA = "e10c513fdbecb27e08319c462ba1d1020b1c94c4ff5d9199047ae513197dd959"
AUTH_RESULT_SHA = "a5b8e70cd40e241e16a250347cf06258a6540100f40423bc7216cb3639191265"
R7C_AUDIT_SHA = "fc1c3d0b6eb1465e147e4a22f0ef8eaeb2095d5123079407a0996018caff5864"
AUTH_CHECKS = {"authorization_only_runner", "fixed_verifier_unchanged", "hash_bindings", "no_device_static", "open_exact", "output_absent", "r7p_pass18"}
OUTER_STAGES = ["psutil_import", "start_ram", "payload", "post_payload_resource", "predevice", "device_execute", "serialize_commit"]
CHAIN = {
    "runner_sha256": S / "run_het_next_l0_ph1_intel_execution_r7c1.py",
    "verifier_sha256": Path(__file__),
    "preflight_sha256": S / "preflight_het_next_l0_ph1_intel_execution_r7c1.py",
    "prereg_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R7C1_PREREGISTRATION_2026-08-14.md",
    "r7c_runner_sha256": S / "run_het_next_l0_ph1_intel_execution_r7c.py",
    "r7c_verifier_sha256": S / "verify_het_next_l0_ph1_intel_execution_r7c.py",
    "r7c_preflight_sha256": S / "preflight_het_next_l0_ph1_intel_execution_r7c.py",
    "r7c_prereg_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R7C_PREREGISTRATION_2026-08-14.md",
    "r7c_lock_sha256": R / "het_next_l0_ph1_intel_execution_r7c_lock.json",
    "r7c_audit_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R7C_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md",
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
    "backend_sha256": S / "het_next_l0_ph1_intel_execution_r6_backend.py",
    "common_sha256": S / "het_next_l0_ph1_intel_execution_r6_common.py",
    "r0_backend_sha256": S / "het_next_l0_ph1_intel_execution_r0_backend.py",
    "r0_runner_sha256": S / "run_het_next_l0_ph1_intel_execution_r0.py",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def auth_result_valid(result: dict) -> bool:
    checks = result.get("checks")
    return (
        set(result) == {"ack", "checks", "kind", "no_payload_compiler_device", "pass", "passed", "r7p_result_sha256", "total"}
        and result.get("kind") == "ph1_intel_execution_r7a_authorization_preflight"
        and result.get("pass") is True and result.get("passed") == result.get("total") == 7
        and result.get("no_payload_compiler_device") is True and result.get("ack") == R7A_ACK
        and result.get("r7p_result_sha256") == R7P_SHA and isinstance(checks, dict)
        and set(checks) == AUTH_CHECKS and all(value is True for value in checks.values())
    )


def extension_valid(extension: dict, lock: dict, observed: dict, auth_result: dict) -> bool:
    return (
        set(lock) == {"kind", "execution_open", "audit_token", "physical_output", "physical_verifier", *observed.keys()}
        and lock.get("kind") == "ph1_intel_execution_r7c1_lock"
        and lock.get("execution_open") is True and lock.get("audit_token") == ACK
        and lock.get("physical_output") == "het_next_l0_ph1_intel_execution_r7a"
        and lock.get("physical_verifier") == "verify_het_next_l0_ph1_intel_execution_r7c1.py"
        and all(lock.get(name) == digest for name, digest in observed.items())
        and set(extension) == {"lock_sha256", "observed", "authorization_result_sha256", "authorization_result", "audit_token", "outer_failure_stages"}
        and extension.get("lock_sha256") == sha256(LOCK) and extension.get("observed") == observed
        and extension.get("authorization_result_sha256") == AUTH_RESULT_SHA
        and extension.get("authorization_result") == auth_result and extension.get("audit_token") == ACK
        and extension.get("outer_failure_stages") == OUTER_STAGES
        and observed.get("authorization_result_sha256") == AUTH_RESULT_SHA
        and observed.get("r7c_audit_sha256") == R7C_AUDIT_SHA
        and observed.get("r7p_result_sha256") == R7P_SHA and auth_result_valid(auth_result)
    )


def main() -> int:
    result_path, manifest_path, commit_path = (OUT / name for name in ("result.json", "manifest.json", "commit.json"))
    result_bytes = result_path.read_bytes(); result = json.loads(result_bytes)
    lock = json.loads(LOCK.read_text()); auth_result = json.loads(AUTH_RESULT.read_text())
    observed = {name: sha256(path) for name, path in CHAIN.items()}
    extension = result.get("authorization", {}).get("r7c1_authorization", {})
    extension_pass = sha256(AUTH_RESULT) == AUTH_RESULT_SHA and extension_valid(extension, lock, observed, auth_result)
    checks = {"r7c1_authorization_extension": extension_pass}
    if extension_pass:
        if observed["physical_verifier_sha256"] != "18b64765469e38c5211d28afe586e0a559e97f6e2110f09f54c4f58d9c38dd88":
            raise RuntimeError("physical_verifier_hash")
        sys.path.insert(0, str(S))
        import verify_het_next_l0_ph1_intel_execution_r7a as numerical
        checks.update(numerical.verify_dict(result))
        manifest = json.loads(manifest_path.read_text())
        checks["bundle"] = numerical.verify_bundle_contract(result_bytes, manifest, json.loads(commit_path.read_text()), {path.name for path in OUT.iterdir()}, sum(path.stat().st_size for path in OUT.iterdir()))
    output = {"kind": "ph1_intel_execution_r7c1_independent_verification", "checks": checks, "pass": all(checks.values()), "passed": sum(value is True for value in checks.values()), "total": len(checks), "claim": "one real expert/input Intel correctness component only"}
    if VERIFY.exists():
        raise FileExistsError(VERIFY)
    with VERIFY.open("x") as handle:
        json.dump(output, handle, sort_keys=True, indent=2); handle.write("\n")
    print(json.dumps(output, indent=2)); return 0 if output["pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
