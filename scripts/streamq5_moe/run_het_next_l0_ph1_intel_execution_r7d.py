#!/usr/bin/env python3
"""PH1 Intel R7D: final auth-only gate over the audited R7C2 lifecycle."""
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
import run_het_next_l0_ph1_intel_execution_r7c2 as lifecycle

LOCK = R / "het_next_l0_ph1_intel_execution_r7d_lock.json"
PREREG = R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R7D_PREREGISTRATION_2026-08-14.md"
VERIFIER = S / "verify_het_next_l0_ph1_intel_execution_r7d.py"
R7C2_RESULT = R / "het_next_l0_ph1_intel_execution_r7c2_static_preflight.json"
R7A_RESULT = R / "het_next_l0_ph1_intel_execution_r7a_authorization_preflight.json"
R7P_RESULT = R / "het_next_l0_ph1_intel_execution_r7p_static_preflight.json"
FAILED = R / "het_next_l0_ph1_intel_execution_r7d_failed_attempts"
QUARANTINE = R / "het_next_l0_ph1_intel_execution_r7d_quarantine"
REVISION_OUT = R / "het_next_l0_ph1_intel_execution_r7d"
VERIFY_RESULT = R / "het_next_l0_ph1_intel_execution_r7d_independent_verification.json"
ACK = "PH1_INTEL_EXECUTION_R7D_AFTER_R7C2_PASS9_AND_FINAL_AUDIT_GO"
R7C2_SHA = "de8745c02cd0b2951adbb04338cf350704608023530edd91a260b73880ebcd8c"
R7C2_AUDIT_SHA = "d6f9ca23a43bef30c0a907efa7997f2d755e6fa3a32c0ed5dab1c21498863a5e"
R7A_SHA = "a5b8e70cd40e241e16a250347cf06258a6540100f40423bc7216cb3639191265"
R7P_SHA = "e10c513fdbecb27e08319c462ba1d1020b1c94c4ff5d9199047ae513197dd959"
R7C2_CHECKS = {"auth_result_exact", "candidate_import_free", "clean_state_complete", "closed_pending", "device_opened_exact", "extension_mutations", "hash_bindings", "inherited_r7c1_cases", "no_device_static"}
R7A_CHECKS = {"authorization_only_runner", "fixed_verifier_unchanged", "hash_bindings", "no_device_static", "open_exact", "output_absent", "r7p_pass18"}
R7P_MUTATIONS = ("getinfo_status", "setptr_status", "ownership_missing", "ownership_duplicate", "ownership_return", "ownership_pending", "ownership_pointer", "identity", "control_missing", "output", "pointer_alias", "alignment", "usm_type", "usm_base", "arg_pointer", "launch_geometry", "launch_event", "read_order", "release_order", "release_owned", "release_code", "cleanup", "provenance", "resource_summary", "resource_order", "resource_peak", "forbidden_api", "stage_hash")
CHAIN = {
    "runner_sha256": Path(__file__),
    "verifier_sha256": VERIFIER,
    "prereg_sha256": PREREG,
    "r7c2_runner_sha256": S / "run_het_next_l0_ph1_intel_execution_r7c2.py",
    "r7c2_verifier_sha256": S / "verify_het_next_l0_ph1_intel_execution_r7c2.py",
    "r7c2_preflight_sha256": S / "preflight_het_next_l0_ph1_intel_execution_r7c2.py",
    "r7c2_prereg_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R7C2_PREREGISTRATION_2026-08-14.md",
    "r7c2_lock_sha256": R / "het_next_l0_ph1_intel_execution_r7c2_lock.json",
    "r7c2_audit_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R7C2_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md",
    "r7c2_result_sha256": R7C2_RESULT,
    **{name: path for name, path in lifecycle.CHAIN.items() if name not in {"runner_sha256", "verifier_sha256", "preflight_sha256", "prereg_sha256"}},
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
    linear_ok = set(linear) == {"gate_up", "down"} and all(row.get("all_rows_equal") is True and row.get("repeat_equal") is True and row.get("first_sha256") == row.get("expected_sha256") == row.get("second_sha256") for row in linear.values())
    negative_ok = negative.get("pass") is True and negative.get("r7_assignment_inside_row_loop") is True and negative.get("mutant_assignment_inside_row_loop") is False and all(row.get("poison_prefix") is True and row.get("last_correct") is True and row.get("repeat_equal") is True and row.get("differs_from_target") is True for row in negative.get("shapes", {}).values())
    return result.get("kind") == "ph1_intel_execution_r7p_static_preflight" and result.get("pass") is True and result.get("passed") == result.get("total") == 18 and result.get("no_payload_compiler_device") is True and len(checks) == 18 and all(value is True for value in checks.values()) and len(baseline) == 20 and all(value is True for value in baseline.values()) and fixture.get("baseline_false_names") == [] and tuple(fixture.get("mutation_names", ())) == R7P_MUTATIONS == tuple(fixture.get("rejected_mutations", ())) and linear_ok and negative_ok


def r7c2_pass9(result: dict) -> bool:
    checks = result.get("checks"); inherited = result.get("inherited_cases"); device = result.get("device_opened_cases"); clean = result.get("clean_state", {})
    return set(result) == {"checks", "clean_state", "device_opened_cases", "inherited_cases", "kind", "no_payload_compiler_device", "pass", "passed", "rejected_extension_mutations", "total"} and result.get("kind") == "ph1_intel_execution_r7c2_static_preflight" and result.get("pass") is True and result.get("passed") == result.get("total") == 9 and result.get("no_payload_compiler_device") is True and isinstance(checks, dict) and set(checks) == R7C2_CHECKS and all(value is True for value in checks.values()) and isinstance(inherited, dict) and len(inherited) == 9 and all(value is True for value in inherited.values()) and device == {"false_retained": True, "true_retained": True, "wrong_type_rejected": True} and result.get("rejected_extension_mutations") == ["token", "result_hash", "check_false", "observed_missing", "lock_hash", "lock_closed", "r7p", "stages"] and all(value is True for section in (clean.get("required", {}), clean.get("absent", {}), clean.get("glob_empty", {})) for value in section.values()) and all(paths == [] for paths in clean.get("glob_paths", {}).values())


def clean_now() -> bool:
    absent = (physical.OUT, physical.FAILED, physical.QUAR, REVISION_OUT, FAILED, QUARANTINE, VERIFY_RESULT)
    stale = [path for path in R.glob("*.inprogress") if any(tag in path.name for tag in ("r7a", "r7d"))]
    return all(not path.exists() for path in absent) and stale == []


def authorize() -> dict:
    if sha256(R7C2_RESULT) != R7C2_SHA or sha256(R7A_RESULT) != R7A_SHA or sha256(R7P_RESULT) != R7P_SHA: raise RuntimeError("preflight_hash")
    r7c2 = json.loads(R7C2_RESULT.read_text()); r7a = json.loads(R7A_RESULT.read_text()); r7p = json.loads(R7P_RESULT.read_text())
    if not (r7c2_pass9(r7c2) and r7a_pass7(r7a) and r7p_pass18(r7p)): raise RuntimeError("preflight_contract")
    observed = {name: sha256(path) for name, path in CHAIN.items()}; lock = json.loads(LOCK.read_text())
    if not (set(lock) == {"kind", "execution_open", "audit_token", "physical_output", "physical_verifier", *observed.keys()} and lock.get("kind") == "ph1_intel_execution_r7d_lock" and lock.get("execution_open") is True and lock.get("audit_token") == ACK and lock.get("physical_output") == "het_next_l0_ph1_intel_execution_r7a" and lock.get("physical_verifier") == "verify_het_next_l0_ph1_intel_execution_r7d.py" and all(lock.get(name) == digest for name, digest in observed.items()) and observed["r7c2_result_sha256"] == R7C2_SHA and observed["r7c2_audit_sha256"] == R7C2_AUDIT_SHA and observed["authorization_result_sha256"] == R7A_SHA and observed["r7p_result_sha256"] == R7P_SHA and clean_now()): raise RuntimeError("r7d_authorization")
    inherited = physical.authorize()
    inherited["r7d_authorization"] = {"lock_sha256": sha256(LOCK), "observed": observed, "r7c2_preflight_sha256": R7C2_SHA, "r7c2_preflight": r7c2, "r7a_preflight_sha256": R7A_SHA, "r7a_preflight": r7a, "r7p_preflight_sha256": R7P_SHA, "r7p_preflight": r7p, "audit_token": ACK}
    return inherited


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--ack", required=True); args = parser.parse_args()
    if args.ack != ACK: return 3
    try: authorization = authorize()
    except Exception: return 3
    lifecycle.OUTER_FAILED = FAILED; lifecycle.OUTER_QUARANTINE = QUARANTINE; lifecycle.REVISION_OUT = REVISION_OUT
    return lifecycle.outer_execute(authorization)


if __name__ == "__main__": raise SystemExit(main())
