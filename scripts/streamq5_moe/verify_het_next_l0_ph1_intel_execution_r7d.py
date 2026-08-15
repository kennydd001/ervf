#!/usr/bin/env python3
"""Standalone final R7D authorization-chain and physical-result verifier."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
S = ROOT / "scripts/streamq5_moe"
R = ROOT / "reports/streamq5_moe"
OUT = R / "het_next_l0_ph1_intel_execution_r7a"
VERIFY = R / "het_next_l0_ph1_intel_execution_r7d_independent_verification.json"
LOCK = R / "het_next_l0_ph1_intel_execution_r7d_lock.json"
R7C2_RESULT = R / "het_next_l0_ph1_intel_execution_r7c2_static_preflight.json"
R7A_RESULT = R / "het_next_l0_ph1_intel_execution_r7a_authorization_preflight.json"
R7P_RESULT = R / "het_next_l0_ph1_intel_execution_r7p_static_preflight.json"
FAILED = R / "het_next_l0_ph1_intel_execution_r7d_failed_attempts"
QUARANTINE = R / "het_next_l0_ph1_intel_execution_r7d_quarantine"
REVISION_OUT = R / "het_next_l0_ph1_intel_execution_r7d"
ACK = "PH1_INTEL_EXECUTION_R7D_AFTER_R7C2_PASS9_AND_FINAL_AUDIT_GO"
R7C2_SHA = "de8745c02cd0b2951adbb04338cf350704608023530edd91a260b73880ebcd8c"
R7C2_AUDIT_SHA = "d6f9ca23a43bef30c0a907efa7997f2d755e6fa3a32c0ed5dab1c21498863a5e"
R7A_SHA = "a5b8e70cd40e241e16a250347cf06258a6540100f40423bc7216cb3639191265"
R7P_SHA = "e10c513fdbecb27e08319c462ba1d1020b1c94c4ff5d9199047ae513197dd959"
R7C2_CHECKS = {"auth_result_exact", "candidate_import_free", "clean_state_complete", "closed_pending", "device_opened_exact", "extension_mutations", "hash_bindings", "inherited_r7c1_cases", "no_device_static"}
R7A_CHECKS = {"authorization_only_runner", "fixed_verifier_unchanged", "hash_bindings", "no_device_static", "open_exact", "output_absent", "r7p_pass18"}
R7P_MUTATIONS = ("getinfo_status", "setptr_status", "ownership_missing", "ownership_duplicate", "ownership_return", "ownership_pending", "ownership_pointer", "identity", "control_missing", "output", "pointer_alias", "alignment", "usm_type", "usm_base", "arg_pointer", "launch_geometry", "launch_event", "read_order", "release_order", "release_owned", "release_code", "cleanup", "provenance", "resource_summary", "resource_order", "resource_peak", "forbidden_api", "stage_hash")
CHAIN = {
    "runner_sha256": S / "run_het_next_l0_ph1_intel_execution_r7d.py",
    "verifier_sha256": Path(__file__),
    "prereg_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R7D_PREREGISTRATION_2026-08-14.md",
    "r7c2_runner_sha256": S / "run_het_next_l0_ph1_intel_execution_r7c2.py",
    "r7c2_verifier_sha256": S / "verify_het_next_l0_ph1_intel_execution_r7c2.py",
    "r7c2_preflight_sha256": S / "preflight_het_next_l0_ph1_intel_execution_r7c2.py",
    "r7c2_prereg_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R7C2_PREREGISTRATION_2026-08-14.md",
    "r7c2_lock_sha256": R / "het_next_l0_ph1_intel_execution_r7c2_lock.json",
    "r7c2_audit_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R7C2_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md",
    "r7c2_result_sha256": R7C2_RESULT,
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
    "authorization_result_sha256": R7A_RESULT,
    "r7a_audit_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R7A_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md",
    "r7p_preflight_sha256": S / "preflight_het_next_l0_ph1_intel_execution_r7p.py",
    "r7p_lock_sha256": R / "het_next_l0_ph1_intel_execution_r7p_lock.json",
    "r7p_prereg_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R7P_PREREGISTRATION_2026-08-14.md",
    "r7p_audit_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R7P_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md",
    "r7p_result_sha256": R7P_RESULT,
    "backend_sha256": S / "het_next_l0_ph1_intel_execution_r6_backend.py",
    "common_sha256": S / "het_next_l0_ph1_intel_execution_r6_common.py",
    "r0_backend_sha256": S / "het_next_l0_ph1_intel_execution_r0_backend.py",
    "r0_runner_sha256": S / "run_het_next_l0_ph1_intel_execution_r0.py",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def r7a_pass7(result: dict) -> bool:
    checks = result.get("checks")
    return set(result) == {"ack", "checks", "kind", "no_payload_compiler_device", "pass", "passed", "r7p_result_sha256", "total"} and result.get("kind") == "ph1_intel_execution_r7a_authorization_preflight" and result.get("pass") is True and result.get("passed") == result.get("total") == 7 and result.get("no_payload_compiler_device") is True and result.get("r7p_result_sha256") == R7P_SHA and isinstance(checks, dict) and set(checks) == R7A_CHECKS and all(value is True for value in checks.values())


def r7p_pass18(result: dict) -> bool:
    checks = result.get("checks", {}); fixture = result.get("verifier_fixture_evidence", {}); baseline = fixture.get("baseline_checks", {}); linear = result.get("linear_sentinel", {}).get("shapes", {}); negative = result.get("write_after_loop_negative", {})
    return result.get("kind") == "ph1_intel_execution_r7p_static_preflight" and result.get("pass") is True and result.get("passed") == result.get("total") == 18 and result.get("no_payload_compiler_device") is True and len(checks) == 18 and all(value is True for value in checks.values()) and len(baseline) == 20 and all(value is True for value in baseline.values()) and fixture.get("baseline_false_names") == [] and tuple(fixture.get("mutation_names", ())) == R7P_MUTATIONS == tuple(fixture.get("rejected_mutations", ())) and set(linear) == {"gate_up", "down"} and all(row.get("all_rows_equal") is True and row.get("repeat_equal") is True and row.get("first_sha256") == row.get("expected_sha256") == row.get("second_sha256") for row in linear.values()) and negative.get("pass") is True and negative.get("r7_assignment_inside_row_loop") is True and negative.get("mutant_assignment_inside_row_loop") is False and all(row.get("poison_prefix") is True and row.get("last_correct") is True and row.get("repeat_equal") is True and row.get("differs_from_target") is True for row in negative.get("shapes", {}).values())


def r7c2_pass9(result: dict) -> bool:
    checks = result.get("checks"); inherited = result.get("inherited_cases"); clean = result.get("clean_state", {})
    return set(result) == {"checks", "clean_state", "device_opened_cases", "inherited_cases", "kind", "no_payload_compiler_device", "pass", "passed", "rejected_extension_mutations", "total"} and result.get("kind") == "ph1_intel_execution_r7c2_static_preflight" and result.get("pass") is True and result.get("passed") == result.get("total") == 9 and result.get("no_payload_compiler_device") is True and isinstance(checks, dict) and set(checks) == R7C2_CHECKS and all(value is True for value in checks.values()) and isinstance(inherited, dict) and len(inherited) == 9 and all(value is True for value in inherited.values()) and result.get("device_opened_cases") == {"false_retained": True, "true_retained": True, "wrong_type_rejected": True} and result.get("rejected_extension_mutations") == ["token", "result_hash", "check_false", "observed_missing", "lock_hash", "lock_closed", "r7p", "stages"] and all(value is True for section in (clean.get("required", {}), clean.get("absent", {}), clean.get("glob_empty", {})) for value in section.values()) and all(paths == [] for paths in clean.get("glob_paths", {}).values())


def extension_valid(extension: dict, lock: dict, observed: dict, r7c2: dict, r7a: dict, r7p: dict) -> bool:
    return set(lock) == {"kind", "execution_open", "audit_token", "physical_output", "physical_verifier", *observed.keys()} and lock.get("kind") == "ph1_intel_execution_r7d_lock" and lock.get("execution_open") is True and lock.get("audit_token") == ACK and lock.get("physical_output") == "het_next_l0_ph1_intel_execution_r7a" and lock.get("physical_verifier") == "verify_het_next_l0_ph1_intel_execution_r7d.py" and all(lock.get(name) == digest for name,digest in observed.items()) and set(extension) == {"lock_sha256", "observed", "r7c2_preflight_sha256", "r7c2_preflight", "r7a_preflight_sha256", "r7a_preflight", "r7p_preflight_sha256", "r7p_preflight", "audit_token"} and extension.get("lock_sha256") == sha256(LOCK) and extension.get("observed") == observed and extension.get("r7c2_preflight_sha256") == R7C2_SHA and extension.get("r7c2_preflight") == r7c2 and extension.get("r7a_preflight_sha256") == R7A_SHA and extension.get("r7a_preflight") == r7a and extension.get("r7p_preflight_sha256") == R7P_SHA and extension.get("r7p_preflight") == r7p and extension.get("audit_token") == ACK and observed.get("r7c2_result_sha256") == R7C2_SHA and observed.get("r7c2_audit_sha256") == R7C2_AUDIT_SHA and observed.get("authorization_result_sha256") == R7A_SHA and observed.get("r7p_result_sha256") == R7P_SHA and r7c2_pass9(r7c2) and r7a_pass7(r7a) and r7p_pass18(r7p)


def main() -> int:
    result_path, manifest_path, commit_path = (OUT / name for name in ("result.json", "manifest.json", "commit.json")); result_bytes = result_path.read_bytes(); result = json.loads(result_bytes); lock = json.loads(LOCK.read_text()); r7c2 = json.loads(R7C2_RESULT.read_text()); r7a = json.loads(R7A_RESULT.read_text()); r7p = json.loads(R7P_RESULT.read_text()); observed = {name: sha256(path) for name,path in CHAIN.items()}; extension = result.get("authorization", {}).get("r7d_authorization", {})
    extension_pass = sha256(R7C2_RESULT) == R7C2_SHA and sha256(R7A_RESULT) == R7A_SHA and sha256(R7P_RESULT) == R7P_SHA and extension_valid(extension, lock, observed, r7c2, r7a, r7p); checks = {"r7d_authorization_extension": extension_pass}
    if extension_pass:
        if observed["physical_verifier_sha256"] != "18b64765469e38c5211d28afe586e0a559e97f6e2110f09f54c4f58d9c38dd88": raise RuntimeError("physical_verifier_hash")
        sys.path.insert(0, str(S)); import verify_het_next_l0_ph1_intel_execution_r7a as numerical
        checks.update(numerical.verify_dict(result)); manifest = json.loads(manifest_path.read_text()); checks["bundle"] = numerical.verify_bundle_contract(result_bytes, manifest, json.loads(commit_path.read_text()), {path.name for path in OUT.iterdir()}, sum(path.stat().st_size for path in OUT.iterdir()))
        checks["r7d_lifecycle_clean"] = not REVISION_OUT.exists() and not FAILED.exists() and not QUARANTINE.exists() and not any(path for path in R.glob("*.inprogress") if "r7d" in path.name)
    output = {"kind": "ph1_intel_execution_r7d_independent_verification", "checks": checks, "pass": all(checks.values()), "passed": sum(value is True for value in checks.values()), "total": len(checks), "claim": "one real expert/input Intel correctness component only"}
    if VERIFY.exists(): raise FileExistsError(VERIFY)
    with VERIFY.open("x") as handle: json.dump(output,handle,sort_keys=True,indent=2); handle.write("\n")
    print(json.dumps(output,indent=2)); return 0 if output["pass"] else 3


if __name__ == "__main__": raise SystemExit(main())
