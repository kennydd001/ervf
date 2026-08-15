#!/usr/bin/env python3
"""PH1 Intel R7D1: final one-path clean-state authorization repair."""
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

import run_het_next_l0_ph1_intel_execution_r7d as prior

LOCK = R / "het_next_l0_ph1_intel_execution_r7d1_lock.json"
PREREG = R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R7D1_PREREGISTRATION_2026-08-14.md"
VERIFIER = S / "verify_het_next_l0_ph1_intel_execution_r7d1.py"
R7A_VERIFICATION = R / "het_next_l0_ph1_intel_execution_r7a_independent_verification.json"
FAILED = R / "het_next_l0_ph1_intel_execution_r7d1_failed_attempts"
QUARANTINE = R / "het_next_l0_ph1_intel_execution_r7d1_quarantine"
REVISION_OUT = R / "het_next_l0_ph1_intel_execution_r7d1"
VERIFY_RESULT = R / "het_next_l0_ph1_intel_execution_r7d1_independent_verification.json"
ACK = "PH1_INTEL_EXECUTION_R7D1_AFTER_R7A_VERIFIER_ABSENCE_AUDIT_GO"
CHAIN = {
    "runner_sha256": Path(__file__),
    "verifier_sha256": VERIFIER,
    "prereg_sha256": PREREG,
    "r7d_runner_sha256": S / "run_het_next_l0_ph1_intel_execution_r7d.py",
    "r7d_verifier_sha256": S / "verify_het_next_l0_ph1_intel_execution_r7d.py",
    "r7d_prereg_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R7D_PREREGISTRATION_2026-08-14.md",
    "r7d_lock_sha256": R / "het_next_l0_ph1_intel_execution_r7d_lock.json",
    "r7d_audit_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R7D_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md",
    **{name: path for name, path in prior.CHAIN.items() if name not in {"runner_sha256", "verifier_sha256", "prereg_sha256"}},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def clean_now() -> bool:
    absent = (R7A_VERIFICATION, REVISION_OUT, FAILED, QUARANTINE, VERIFY_RESULT)
    stale = [path for path in R.glob("*.inprogress") if "r7d1" in path.name]
    return all(not path.exists() for path in absent) and stale == [] and prior.clean_now()


def authorize() -> dict:
    if not clean_now(): raise RuntimeError("r7d1_clean_state")
    observed = {name: sha256(path) for name, path in CHAIN.items()}; lock = json.loads(LOCK.read_text())
    if not (set(lock) == {"kind", "execution_open", "audit_token", "physical_output", "physical_verifier", *observed.keys()} and lock.get("kind") == "ph1_intel_execution_r7d1_lock" and lock.get("execution_open") is True and lock.get("audit_token") == ACK and lock.get("physical_output") == "het_next_l0_ph1_intel_execution_r7a" and lock.get("physical_verifier") == "verify_het_next_l0_ph1_intel_execution_r7d1.py" and all(lock.get(name) == digest for name,digest in observed.items()) and observed["r7d_audit_sha256"] == "8f798ac7b5f4d98e195ac076f54aaf988c927c51cbb76d97ce19b46e72f0182f" and observed["r7c2_result_sha256"] == prior.R7C2_SHA): raise RuntimeError("r7d1_authorization")
    inherited = prior.authorize()
    inherited["r7d1_authorization"] = {"lock_sha256": sha256(LOCK), "observed": observed, "r7a_verification_path": R7A_VERIFICATION.name, "r7a_verification_absent": True, "audit_token": ACK}
    return inherited


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--ack", required=True); args = parser.parse_args()
    if args.ack != ACK: return 3
    try: authorization = authorize()
    except Exception: return 3
    prior.lifecycle.OUTER_FAILED = FAILED; prior.lifecycle.OUTER_QUARANTINE = QUARANTINE; prior.lifecycle.REVISION_OUT = REVISION_OUT
    return prior.lifecycle.outer_execute(authorization)


if __name__ == "__main__": raise SystemExit(main())
