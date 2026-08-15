#!/usr/bin/env python3
"""Standalone R7D1 one-path clean-state and physical-result verifier."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
S = ROOT / "scripts/streamq5_moe"
R = ROOT / "reports/streamq5_moe"
OUT = R / "het_next_l0_ph1_intel_execution_r7a"
VERIFY = R / "het_next_l0_ph1_intel_execution_r7d1_independent_verification.json"
LOCK = R / "het_next_l0_ph1_intel_execution_r7d1_lock.json"
R7A_VERIFICATION = R / "het_next_l0_ph1_intel_execution_r7a_independent_verification.json"
FAILED = R / "het_next_l0_ph1_intel_execution_r7d1_failed_attempts"
QUARANTINE = R / "het_next_l0_ph1_intel_execution_r7d1_quarantine"
REVISION_OUT = R / "het_next_l0_ph1_intel_execution_r7d1"
ACK = "PH1_INTEL_EXECUTION_R7D1_AFTER_R7A_VERIFIER_ABSENCE_AUDIT_GO"
R7D_AUDIT_SHA = "8f798ac7b5f4d98e195ac076f54aaf988c927c51cbb76d97ce19b46e72f0182f"
CHAIN = {
    "runner_sha256": S / "run_het_next_l0_ph1_intel_execution_r7d1.py",
    "verifier_sha256": Path(__file__),
    "prereg_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R7D1_PREREGISTRATION_2026-08-14.md",
    "r7d_runner_sha256": S / "run_het_next_l0_ph1_intel_execution_r7d.py",
    "r7d_verifier_sha256": S / "verify_het_next_l0_ph1_intel_execution_r7d.py",
    "r7d_prereg_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R7D_PREREGISTRATION_2026-08-14.md",
    "r7d_lock_sha256": R / "het_next_l0_ph1_intel_execution_r7d_lock.json",
    "r7d_audit_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R7D_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md",
    "r7c2_runner_sha256": S / "run_het_next_l0_ph1_intel_execution_r7c2.py",
    "r7c2_verifier_sha256": S / "verify_het_next_l0_ph1_intel_execution_r7c2.py",
    "r7c2_preflight_sha256": S / "preflight_het_next_l0_ph1_intel_execution_r7c2.py",
    "r7c2_prereg_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R7C2_PREREGISTRATION_2026-08-14.md",
    "r7c2_lock_sha256": R / "het_next_l0_ph1_intel_execution_r7c2_lock.json",
    "r7c2_audit_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R7C2_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md",
    "r7c2_result_sha256": R / "het_next_l0_ph1_intel_execution_r7c2_static_preflight.json",
    "r7c1_runner_sha256": S / "run_het_next_l0_ph1_intel_execution_r7c1.py",
    "r7c1_verifier_sha256": S / "verify_het_next_l0_ph1_intel_execution_r7c1.py",
    "r7c1_preflight_sha256": S / "preflight_het_next_l0_ph1_intel_execution_r7c1.py",
    "r7c1_prereg_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R7C1_PREREGISTRATION_2026-08-14.md",
    "r7c1_lock_sha256": R / "het_next_l0_ph1_intel_execution_r7c1_lock.json",
    "r7c1_audit_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R7C1_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md",
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
    "authorization_result_sha256": R / "het_next_l0_ph1_intel_execution_r7a_authorization_preflight.json",
    "r7a_audit_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R7A_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md",
    "r7p_preflight_sha256": S / "preflight_het_next_l0_ph1_intel_execution_r7p.py",
    "r7p_lock_sha256": R / "het_next_l0_ph1_intel_execution_r7p_lock.json",
    "r7p_prereg_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R7P_PREREGISTRATION_2026-08-14.md",
    "r7p_audit_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R7P_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md",
    "r7p_result_sha256": R / "het_next_l0_ph1_intel_execution_r7p_static_preflight.json",
    "backend_sha256": S / "het_next_l0_ph1_intel_execution_r6_backend.py",
    "common_sha256": S / "het_next_l0_ph1_intel_execution_r6_common.py",
    "r0_backend_sha256": S / "het_next_l0_ph1_intel_execution_r0_backend.py",
    "r0_runner_sha256": S / "run_het_next_l0_ph1_intel_execution_r0.py"
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def extension_valid(extension: dict, lock: dict, observed: dict) -> bool:
    return set(lock) == {"kind", "execution_open", "audit_token", "physical_output", "physical_verifier", *observed.keys()} and lock.get("kind") == "ph1_intel_execution_r7d1_lock" and lock.get("execution_open") is True and lock.get("audit_token") == ACK and lock.get("physical_output") == "het_next_l0_ph1_intel_execution_r7a" and lock.get("physical_verifier") == "verify_het_next_l0_ph1_intel_execution_r7d1.py" and all(lock.get(name) == digest for name,digest in observed.items()) and set(extension) == {"lock_sha256", "observed", "r7a_verification_path", "r7a_verification_absent", "audit_token"} and extension.get("lock_sha256") == sha256(LOCK) and extension.get("observed") == observed and extension.get("r7a_verification_path") == R7A_VERIFICATION.name and extension.get("r7a_verification_absent") is True and extension.get("audit_token") == ACK and observed.get("r7d_audit_sha256") == R7D_AUDIT_SHA


def main() -> int:
    result_path, manifest_path, commit_path = (OUT / name for name in ("result.json", "manifest.json", "commit.json")); result_bytes = result_path.read_bytes(); result = json.loads(result_bytes); observed = {name: sha256(path) for name,path in CHAIN.items()}; lock = json.loads(LOCK.read_text()); extension = result.get("authorization", {}).get("r7d1_authorization", {})
    extension_pass = not R7A_VERIFICATION.exists() and extension_valid(extension, lock, observed); checks = {"r7d1_authorization_extension": extension_pass}
    if extension_pass:
        if observed["r7d_verifier_sha256"] != "8fa44558412eed80891d013fda8a08881e65ca30caf35062c9b7428a02d10fb4": raise RuntimeError("r7d_verifier_hash")
        sys.path.insert(0, str(S)); import verify_het_next_l0_ph1_intel_execution_r7d as prior_verifier
        r7d_extension = result.get("authorization", {}).get("r7d_authorization", {}); r7c2 = json.loads(prior_verifier.R7C2_RESULT.read_text()); r7a = json.loads(prior_verifier.R7A_RESULT.read_text()); r7p = json.loads(prior_verifier.R7P_RESULT.read_text()); r7d_observed = {name: prior_verifier.sha256(path) for name,path in prior_verifier.CHAIN.items()}; checks["inherited_r7d_authorization"] = prior_verifier.extension_valid(r7d_extension, json.loads(prior_verifier.LOCK.read_text()), r7d_observed, r7c2, r7a, r7p)
        if checks["inherited_r7d_authorization"]:
            if observed["physical_verifier_sha256"] != "18b64765469e38c5211d28afe586e0a559e97f6e2110f09f54c4f58d9c38dd88": raise RuntimeError("physical_verifier_hash")
            import verify_het_next_l0_ph1_intel_execution_r7a as numerical
            checks.update(numerical.verify_dict(result)); manifest = json.loads(manifest_path.read_text()); checks["bundle"] = numerical.verify_bundle_contract(result_bytes, manifest, json.loads(commit_path.read_text()), {path.name for path in OUT.iterdir()}, sum(path.stat().st_size for path in OUT.iterdir()))
            checks["r7d1_lifecycle_clean"] = not REVISION_OUT.exists() and not FAILED.exists() and not QUARANTINE.exists() and not R7A_VERIFICATION.exists() and not any(path for path in R.glob("*.inprogress") if "r7d1" in path.name)
    output = {"kind": "ph1_intel_execution_r7d1_independent_verification", "checks": checks, "pass": all(checks.values()), "passed": sum(value is True for value in checks.values()), "total": len(checks), "claim": "one real expert/input Intel correctness component only"}
    if VERIFY.exists(): raise FileExistsError(VERIFY)
    with VERIFY.open("x") as handle: json.dump(output,handle,sort_keys=True,indent=2); handle.write("\n")
    print(json.dumps(output,indent=2)); return 0 if output["pass"] else 3


if __name__ == "__main__": raise SystemExit(main())
